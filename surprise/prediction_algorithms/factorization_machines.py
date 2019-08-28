"""
the :mod:`fm` module includes a factorization machine algorithm.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy
from collections import defaultdict
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.utils.data

from .predictions import PredictionImpossible
from .algo_base import AlgoBase


class CandidateDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, w):
        self.len = x.shape[0]
        self.x = x
        self.y = y
        self.w = w

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.w[index]

    def __len__(self):
        return self.len


class MyCriterion(nn.Module):
    """ The PyTorch model for the loss function. This class is used by
    `FM`.

    Args:
        sample_weight: bool, default: False
            Whether sample weights are used.
        binary: bool, default: False
            Whether the output is binary.
    """

    def __init__(self, sample_weight=False, binary=False):

        super().__init__()
        self.sample_weight = sample_weight
        self.binary = binary

        if self.binary:
            if self.sample_weight:
                self.loss_fn = nn.BCELoss(reduction='none')
            else:
                self.loss_fn = nn.BCELoss()
        else:
            if self.sample_weight:
                self.loss_fn = nn.MSELoss(reduction='none')
            else:
                self.loss_fn = nn.MSELoss()

    def forward(self, y_pred, y, w):

        loss = self.loss_fn(y_pred, y)

        if self.sample_weight:
            loss = torch.dot(w, loss) / torch.sum(w)

        return loss


class FMMixin():

    def set_random_state(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)

    def _construct_fit_data(self):
        """ Construct the data needed by the `fit()` function.

        It is assumed that the user and item features are correctly encoded.
        These dummies are created (if needed) using only the info in the
        trainset.
        """

        if self.user_lst and (self.trainset.n_user_features == 0):
            raise ValueError('user_lst cannot be used since '
                             'there are no user_features')
        if self.item_lst and (self.trainset.n_item_features == 0):
            raise ValueError('item_lst cannot be used since '
                             'there are no item_features')

        n_ratings = self.trainset.n_ratings
        n_users = self.trainset.n_users
        n_items = self.trainset.n_items

        # Construct ratings_df from trainset
        # The IDs are unique and start at 0
        ratings_df = pd.DataFrame([tup for tup in self.trainset.all_ratings()],
                                  columns=['userID', 'itemID', 'rating'])

        # Initialize df with rating values
        libsvm_df = pd.DataFrame(ratings_df['rating'])

        # Remove offset if binary
        if self.binary:
            libsvm_df['rating'] -= self.trainset.offset

        # Add sample_weight column
        libsvm_df['sample_weight'] = ratings_df.apply(
            lambda x: self.trainset.sample_weight.get(
                (x['userID'], x['itemID']), np.nan), axis=1)

        field_dims = []

        # Add rating features
        if self.rating_lst:
            for feature in self.rating_lst:
                if feature == 'userID':
                    temp = pd.get_dummies(
                        ratings_df['userID'], prefix='userID')
                    libsvm_df = pd.concat([libsvm_df, temp], axis=1)
                    field_dims.append(temp.shape[1])
                elif feature == 'itemID':
                    temp = pd.get_dummies(
                        ratings_df['itemID'], prefix='itemID')
                    libsvm_df = pd.concat([libsvm_df, temp], axis=1)
                    field_dims.append(temp.shape[1])
                elif feature == 'imp_u_rating':
                    temp = np.zeros((n_ratings, n_items))
                    for row in ratings_df.itertuples():
                        iid = row.itemID
                        all_u_ratings = self.trainset.ur[row.userID]
                        for other_iid, rating, w in all_u_ratings:
                            if other_iid != iid:  # only the other ratings
                                temp[row.Index, other_iid] = 1
                    count = np.count_nonzero(temp, axis=1)[:, None]
                    count[count == 0] = 1  # remove zeros for division
                    temp = temp / count
                    cols = ['imp_u_rating_{}'.format(i)
                            for i in range(n_items)]
                    libsvm_df = pd.concat([libsvm_df, pd.DataFrame(
                        temp, columns=cols)], axis=1)
                    field_dims.append(len(cols))
                elif feature == 'exp_u_rating':
                    # a rating is at least 1 with the offset
                    temp = np.zeros((n_ratings, n_items))
                    for row in ratings_df.itertuples():
                        iid = row.itemID
                        all_u_ratings = self.trainset.ur[row.userID]
                        for other_iid, rating, w in all_u_ratings:
                            if other_iid != iid:  # only the other ratings
                                temp[row.Index, other_iid] = rating
                    count = np.count_nonzero(temp, axis=1)[:, None]
                    count[count == 0] = 1  # remove zeros for division
                    temp = temp / count
                    cols = ['exp_u_rating_{}'.format(i)
                            for i in range(n_items)]
                    libsvm_df = pd.concat([libsvm_df, pd.DataFrame(
                        temp, columns=cols)], axis=1)
                    field_dims.append(len(cols))
                elif feature == 'imp_i_rating':
                    temp = np.zeros((n_ratings, n_users))
                    for row in ratings_df.itertuples():
                        uid = row.userID
                        all_i_ratings = self.trainset.ir[row.itemID]
                        for other_uid, rating, w in all_i_ratings:
                            if other_uid != uid:  # only the other ratings
                                temp[row.Index, other_uid] = 1
                    count = np.count_nonzero(temp, axis=1)[:, None]
                    count[count == 0] = 1  # remove zeros for division
                    temp = temp / count
                    cols = ['imp_i_rating_{}'.format(u)
                            for u in range(n_users)]
                    libsvm_df = pd.concat([libsvm_df, pd.DataFrame(
                        temp, columns=cols)], axis=1)
                    field_dims.append(len(cols))
                elif feature == 'exp_i_rating':
                    # a rating is at least 1 with the offset
                    temp = np.zeros((n_ratings, n_users))
                    for row in ratings_df.itertuples():
                        uid = row.userID
                        all_i_ratings = self.trainset.ir[row.itemID]
                        for other_uid, rating, w in all_i_ratings:
                            if other_uid != uid:  # only the other ratings
                                temp[row.Index, other_uid] = rating
                    count = np.count_nonzero(temp, axis=1)[:, None]
                    count[count == 0] = 1  # remove zeros for division
                    temp = temp / count
                    cols = ['exp_i_rating_{}'.format(u)
                            for u in range(n_users)]
                    libsvm_df = pd.concat([libsvm_df, pd.DataFrame(
                        temp, columns=cols)], axis=1)
                    field_dims.append(len(cols))
                else:
                    raise ValueError('{} is not an accepted value '
                                     'for rating_lst'.format(feature))

        # Add user features
        if self.user_lst:
            temp = pd.DataFrame(
                [self.trainset.u_features[uid]
                 for uid in ratings_df['userID']],
                columns=self.trainset.user_features_labels)
            for feature_lst in self.user_lst:
                for feature in feature_lst:
                    if feature in self.trainset.user_features_labels:
                        libsvm_df[feature] = temp[feature]
                    else:
                        raise ValueError(
                            '{} is not part of user_features'.format(feature))
                field_dims.append(len(feature_lst))

        # Add item features
        if self.item_lst:
            temp = pd.DataFrame(
                [self.trainset.i_features[iid]
                 for iid in ratings_df['itemID']],
                columns=self.trainset.item_features_labels)
            for feature_lst in self.item_lst:
                for feature in feature_lst:
                    if feature in self.trainset.item_features_labels:
                        libsvm_df[feature] = temp[feature]
                    else:
                        raise ValueError(
                            '{} is not part of item_features'.format(feature))
                field_dims.append(len(feature_lst))

        self.libsvm_df = libsvm_df
        self.n_features = libsvm_df.shape[1] - 2
        self.field_dims = field_dims

    def _fit(self):

        params = self._add_weight_decay()
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        # self.optimizer = torch.optim.SGD(params, lr=self.lr)

        # Define data
        x = self.libsvm_df.iloc[:, 2:].values.astype('float64')
        y = self.libsvm_df.iloc[:, 0].values.astype('float64')
        if self.trainset.sample_weight:
            sample_weight = True
            w = self.libsvm_df.iloc[:, 1].values.astype('float64')
        else:
            sample_weight = False
            w = np.ones(x.shape[0]).astype('float64')

        # Construct tensors
        x_train, x_dev, y_train, y_dev, w_train, w_dev = train_test_split(
            x, y, w, test_size=self.dev_ratio,
            random_state=self.random_state)
        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)
        w_train = torch.Tensor(w_train)
        x_dev = torch.Tensor(x_dev)
        y_dev = torch.Tensor(y_dev)
        w_dev = torch.Tensor(w_dev)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        w = torch.Tensor(w)

        if self.batch_size is not None:
            # mini batches currently do not work with sparse tensors
            train_data = CandidateDataset(x_train, y_train, w_train)
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.batch_size, shuffle=True)
        elif isinstance(self, FM):
            x_train = x_train.to_sparse()  # take advantage of x sparseness
        if isinstance(self, FM):
            x_dev = x_dev.to_sparse()  # take advantage of x sparseness

        # Construct criterions
        train_criterion = MyCriterion(sample_weight=sample_weight,
                                      binary=self.binary)
        dev_criterion = MyCriterion(sample_weight=sample_weight,
                                    binary=self.binary)

        # Construct train_step
        train_step = FMMixin.make_train_step(self.model,
                                             train_criterion,
                                             self.optimizer)

        # Run training
        best_model = None
        best_loss = np.inf
        best_epoch = None
        counter = 0
        for epoch in range(self.n_epochs):

            if self.batch_size is not None:
                for x_batch, y_batch, w_batch in train_loader:
                    self.train_loss = train_step(x_batch, y_batch, w_batch)
            else:
                self.train_loss = train_step(x_train, y_train, w_train)

            with torch.no_grad():
                # Switch to eval mode and evaluate with development data
                # See https://github.com/pytorch/examples/blob/master/snli/train.py
                self.model.eval()
                y_pred = self.model(x_dev)
                self.dev_loss = dev_criterion(y_pred, y_dev, w_dev).item()

            if self.verbose:
                print(epoch, self.train_loss, self.dev_loss)

            if self.dev_loss < best_loss:
                best_model = copy.deepcopy(self.model)
                best_loss = self.dev_loss
                best_epoch = epoch
                counter = 0
                if self.verbose:
                    print('A new best model have been found!')

            counter += 1

            if counter > self.patience:
                break

        if best_model is None:  # keep last model
            best_model = copy.deepcopy(self.model)
            best_loss = self.dev_loss
            best_epoch = epoch

        if self.refit:
            if self.verbose:
                print('Refitting model with all training data...')
            self.model.initialize()
            params = self._add_weight_decay()
            self.optimizer = torch.optim.Adam(params, lr=self.lr)
            # self.optimizer = torch.optim.SGD(params, lr=self.lr)
            if self.batch_size is not None:
                train_data = CandidateDataset(x, y, w)
                train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=self.batch_size, shuffle=True)
            elif isinstance(self, FM):
                x = x.to_sparse()  # take advantage of x sparseness
            train_step = FMMixin.make_train_step(self.model,
                                                 train_criterion,
                                                 self.optimizer)
            for i in range(best_epoch + 1):
                if self.batch_size is not None:
                    for x_batch, y_batch, w_batch in train_loader:
                        self.train_all_loss = train_step(
                            x_batch, y_batch, w_batch)
                else:
                    self.train_all_loss = train_step(x, y, w)
        else:
            self.model = best_model

        self.best_epoch = best_epoch

    @staticmethod
    def make_train_step(model, criterion, optimizer):
        """ Builds function that performs a step in the train loop.
        """

        def train_step(x, y, w=None):
            model.train()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y, w)
            loss.backward()
            optimizer.step()

            return loss.item()

        return train_step

    def _add_weight_decay(self, skip_list=None):
        """ Add weight_decay with no regularization for bias.
        """

        if skip_list is None:
            skip_list = []

        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if ((len(param.shape) == 1) or name.endswith(".bias") or
                    (name in skip_list)):
                no_decay.append(param)
            else:
                decay.append(param)

        return [{'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': self.reg}]

    def estimate(self, u, i, u_features, i_features):

        torch.set_default_dtype(torch.float64)  # use float64

        # Construct details
        details = {}
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            details['knows_user'] = True
            details['knows_item'] = True
        elif self.trainset.knows_user(u):
            details['knows_user'] = True
            details['knows_item'] = False
        elif self.trainset.knows_item(i):
            details['knows_user'] = False
            details['knows_item'] = True
        else:
            raise PredictionImpossible('Unknown user and item')

        # Estimate rating
        with torch.no_grad():
            self.model.eval()
            x = self._construct_estimate_data(u, i, u_features, i_features)
            x = torch.Tensor(x[None, :])  # add dimension
            est = float(self.model(x))

        # Add offset if binary (since was trained without offset)
        if self.binary:
            est += self.trainset.offset

        return est, details

    def _construct_estimate_data(self, u, i, u_features, i_features):
        """ Construct the data needed by the `estimate()` function.

        It is assumed that if features are given in u_features or i_features,
        they are all given and in the same order as in the trainset.
        """

        if (self.user_lst and u_features and (
                len(u_features) != len(self.trainset.user_features_labels))):
            raise ValueError('If u_features are provided for predict(), they'
                             'should all be provided as in trainset')
        if (self.item_lst and i_features and (
                len(i_features) != len(self.trainset.item_features_labels))):
            raise ValueError('If i_features are provided for predict(), they'
                             'should all be provided as in trainset')

        n_users = self.trainset.n_users
        n_items = self.trainset.n_items

        x = []

        # Add rating features
        if self.rating_lst:
            for feature in self.rating_lst:
                if feature == 'userID':
                    temp = [0.] * n_users
                    if self.trainset.knows_user(u):
                        temp[u] = 1.
                    x.extend(temp)
                elif feature == 'itemID':
                    temp = [0.] * n_items
                    if self.trainset.knows_item(i):
                        temp[i] = 1.
                    x.extend(temp)
                elif feature == 'imp_u_rating':
                    temp = [0.] * n_items
                    if self.trainset.knows_user(u):
                        all_u_ratings = self.trainset.ur[u]
                        for other_i, rating, w in all_u_ratings:
                            if other_i != i:  # only the other ratings
                                temp[other_i] = 1.
                        temp = np.array(temp)
                        count = np.count_nonzero(temp)
                        if count == 0:
                            count = 1
                        temp = list(temp / count)
                    x.extend(temp)
                elif feature == 'exp_u_rating':
                    # a rating is at least 1 with the offset
                    temp = [0.] * n_items
                    if self.trainset.knows_user(u):
                        all_u_ratings = self.trainset.ur[u]
                        for other_i, rating, w in all_u_ratings:
                            if other_i != i:  # only the other ratings
                                temp[other_i] = rating
                        temp = np.array(temp)
                        count = np.count_nonzero(temp)
                        if count == 0:
                            count = 1
                        temp = list(temp / count)
                    x.extend(temp)
                elif feature == 'imp_i_rating':
                    temp = [0.] * n_users
                    if self.trainset.knows_item(i):
                        all_i_ratings = self.trainset.ir[i]
                        for other_u, rating, w in all_i_ratings:
                            if other_u != u:  # only the other ratings
                                temp[other_u] = 1.
                        temp = np.array(temp)
                        count = np.count_nonzero(temp)
                        if count == 0:
                            count = 1
                        temp = list(temp / count)
                    x.extend(temp)
                elif feature == 'exp_i_rating':
                    # a rating is at least 1 with the offset
                    temp = [0.] * n_users
                    if self.trainset.knows_item(i):
                        all_i_ratings = self.trainset.ir[i]
                        for other_u, rating, w in all_i_ratings:
                            if other_u != u:  # only the other ratings
                                temp[other_u] = rating
                        temp = np.array(temp)
                        count = np.count_nonzero(temp)
                        if count == 0:
                            count = 1
                        temp = list(temp / count)
                    x.extend(temp)

        # Add user features
        if self.user_lst:
            feature_lst = [feature for lst in self.user_lst for feature in lst]
            temp = [0.] * len(feature_lst)
            if u_features:
                # It is assumed that if features are given, they are all given.
                temp_df = pd.Series(
                    u_features, index=self.trainset.user_features_labels)
                for idx, feature in enumerate(feature_lst):
                        temp[idx] = temp_df[feature]
            x.extend(temp)

        # Add item features
        if self.item_lst:
            feature_lst = [feature for lst in self.item_lst for feature in lst]
            temp = [0.] * len(feature_lst)
            if i_features:
                # It is assumed that if features are given, they are all given.
                temp_df = pd.Series(
                    i_features, index=self.trainset.item_features_labels)
                for idx, feature in enumerate(feature_lst):
                    temp[idx] = temp_df[feature]
            x.extend(temp)

        return np.array(x).astype('float64')

    def print_model(self):
        """ Print the fitted model.
        """

        if self.model is None:
            print('fit() has not been called; no model to print.')

        for name, param in self.model.named_parameters():
            print(name, param.data)


class FMtorchNN(nn.Module):
    """ The PyTorch model for factorization machine. This class is used by
    `FM`. The initilization is done as in Rendle (2012).

    Args:
        n_features: int
            Defines the number of features in x.
        n_factors: int, default: 20
            Defines the number of factors in the interaction terms.
        init_std: float, default: 0.01
            The standard deviation of the normal distribution for
            initialization.
        binary: bool, default: False
            Whether the output is binary.
    """

    def __init__(self, n_features, n_factors=20, init_std=0.01, binary=False):

        super().__init__()
        self.n_features = n_features
        self.n_factors = n_factors
        self.init_std = init_std
        self.binary = binary

        # Define parameters
        self.b = nn.Parameter(torch.Tensor(1),
                              requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(self.n_features, 1),
                              requires_grad=True)
        self.V = nn.Parameter(torch.Tensor(self.n_features, self.n_factors),
                              requires_grad=True)

        # Initialize parameters
        self.initialize()

        # Define activation if necessary
        if self.binary:
            self.out_act = nn.Sigmoid()

    def initialize(self):

        self.b.data.fill_(0.)
        self.w.data.fill_(0.)
        self.V.data.normal_(0., self.init_std)
        # nn.init.xavier_uniform_(self.w.data)
        # nn.init.xavier_uniform_(self.V.data)
        # nn.init.xavier_normal_(self.w.data)
        # nn.init.xavier_normal_(self.V.data)

    def forward(self, x):

        # The linear part
        total_linear = torch.sum(torch.mm(x, self.w), dim=1)

        # The interaction part
        # O(kn) formulation from Steffen Rendle
        total_inter_1 = torch.mm(x, self.V) ** 2
        total_inter_2 = torch.mm(x ** 2, self.V ** 2)
        total_inter = 0.5 * torch.sum(total_inter_1 - total_inter_2, dim=1)

        # Compute predictions
        y_pred = self.b + total_linear + total_inter
        if self.binary:
            y_pred = self.out_act(y_pred)

        return y_pred


class FM(AlgoBase, FMMixin):
    """A factorization machine algorithm implemented using pytorch.

    Args:
        rating_lst : list of str or `None`, default : ('userID', 'itemID')
            This list specifies what information from the `raw_ratings` to put
            in the `x` vector. Accepted list values are 'userID', 'itemID',
            'imp_u_rating', 'exp_u_rating', 'imp_i_rating' and 'exp_i_rating'.
            Implicit and explicit user/item rating values are scaled by the
            number of values. If `None`, no info is added.
        user_lst : list of str or `None`, default : `None`
            This list specifies what information from the `user_features` to
            put in the `x` vector. Accepted list values consist of the names of
            features. If `None`, no info is added.
        item_lst : list of str or `None`, default : `None`
            This list specifies what information from the `item_features` to
            put in the `x` vector. Accepted list values consist of the names of
            features. If `None`, no info is added.
        n_factors : int, default: 20
            Number of latent factors in low-rank approximation.
        dev_ratio : float, default : 0.3
            Ratio of `trainset` to dedicate to development data set to identify
            best model. Should be either positive and smaller than the number
            of samples or a float in the (0, 1) range.
        patience : int, default : 10
            Number of epochs without improvement to wait before ending
            training. The last best model is kept.
        n_epochs : int, default : 500
            Maximum number of training epochs.
        batch_size: int, default : `None`
            Batch size to use for training. If set to `None`, then training is
            done using full dataset.
        init_std : float, default : 0.01
            The standard deviation of the normal distribution for
            initialization.
        lr : float, default : 0.001
            Learning rate for optimization method.
        reg : float, default : 0.02
            Strength of L2 regularization. It can be disabled by setting it to
            zero.
        refit : bool, default : `True`
            Determines whether to refit model with all training data for the
            best number of epochs found with the development data set.
        binary : bool, default : `False`
            Whether to use a sigmoid on the output to obtain a binary output.
        random_state : int, default : `None`
            Determines the RNG that will be used for initialization. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same initialization over multiple calls to
            ``fit()``. If ``None``, the current RNG from torch is used.
        verbose : bool, default : `False`
            Level of verbosity.
    """

    def __init__(self, rating_lst=('userID', 'itemID'), user_lst=None,
                 item_lst=None, n_factors=20, dev_ratio=0.3, patience=10,
                 n_epochs=500, batch_size=None, init_std=0.01, lr=0.001,
                 reg=0.02, refit=True, binary=False, random_state=None,
                 verbose=False, **kwargs):

        super().__init__(**kwargs)

        # Put user_lst and item_lst within a list (one FFM field)
        if rating_lst is None:
            self.rating_lst = []
        else:
            self.rating_lst = rating_lst
        if user_lst is None:
            self.user_lst = []
        else:
            self.user_lst = [user_lst]
        if item_lst is None:
            self.item_lst = []
        else:
            self.item_lst = [item_lst]
        self.n_factors = n_factors
        self.dev_ratio = dev_ratio
        self.patience = patience
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_std = init_std
        self.lr = lr
        self.reg = reg
        self.refit = refit
        self.binary = binary
        self.random_state = random_state
        self.verbose = verbose

        self.model = None

    def fit(self, trainset):

        # Set pre-requirements
        torch.set_default_dtype(torch.float64)  # use float64
        super().fit(trainset)
        self.set_random_state()

        # Construct data and initialize model
        self._construct_fit_data()
        self.model = FMtorchNN(self.n_features, n_factors=self.n_factors,
                               init_std=self.init_std, binary=self.binary)

        # Train model
        self._fit()

        return self


class FFMtorchNN(nn.Module):
    """ The PyTorch model for field-aware factorization machine. This class is
    used by `FFM`. The initilization is done as in Rendle (2012).

    The field-aware part is only for the factorization.

    Args:
        field_dims: list of int
            Defines (in order) the number of features in each field.
        n_factors: int, default: 20
            Defines the number of factors in the interaction terms.
        init_std: float, default: 0.01
            The standard deviation of the normal distribution for
            initialization.
        binary: bool, default: False
            Whether the output is binary.
    """

    def __init__(self, field_dims, n_factors=20, init_std=0.01, binary=False):

        super().__init__()
        self.field_dims = field_dims
        self.feat2field = [i
                           for i, dim in enumerate(field_dims)
                           for _ in range(dim)]
        self.field2feat = defaultdict(list)
        for feat, field in enumerate(self.feat2field):
            self.field2feat[field].append(feat)
        self.n_fields = len(field_dims)
        self.n_features = sum(field_dims)
        self.n_factors = n_factors
        self.init_std = init_std
        self.binary = binary

        # Define parameters
        self.b = nn.Parameter(torch.Tensor(1),
                              requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(self.n_features, 1),
                              requires_grad=True)
        self.V = nn.Parameter(
            torch.Tensor(self.n_features, self.n_fields, self.n_factors),
            requires_grad=True)

        # Initialize parameters
        self.initialize()

        # Define activation if necessary
        if self.binary:
            self.out_act = nn.Sigmoid()

    def initialize(self):

        self.b.data.fill_(0.)
        self.w.data.fill_(0.)
        self.V.data.normal_(0., self.init_std)

    def forward(self, x):

        # The linear part
        total_linear = torch.sum(torch.mm(x, self.w), dim=1)

        # The interaction part
        # This is clearly not computationally efficient!
        total_inter = torch.zeros(x.shape[0])

        temp_prod = torch.einsum('ij, jkl -> ijkl', x, self.V)
        print(x.shape, self.V.shape, temp_prod.shape)

        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                temp1 = temp_prod[:, i, self.feat2field[j], :]
                temp2 = temp_prod[:, j, self.feat2field[i], :]
                total_inter += 0.5 * torch.sum(temp1 * temp2, dim=1)

        # Compute predictions
        y_pred = self.b + total_linear + total_inter
        if self.binary:
            y_pred = self.out_act(y_pred)
        print('iteration is done')

        return y_pred


class FFM(AlgoBase, FMMixin):
    """A field-aware factorization machine algorithm implemented using pytorch.

    Args:
        rating_lst : list of str or `None`, default : ('userID', 'itemID')
            This list specifies what information from the `raw_ratings` to put
            in the `x` vector. Accepted list values are 'userID', 'itemID',
            'imp_u_rating', 'exp_u_rating', 'imp_i_rating' and 'exp_i_rating'.
            Implicit and explicit user/item rating values are scaled by the
            number of values. If `None`, no info is added.
        user_lst : list of list of str or `None`, default : `None`
            This list specifies what information from the `user_features` to
            put in the `x` vector. Each inner list corresponds to a different
            field. Accepted list values consist of the names of
            features. If `None`, no info is added.
        item_lst : list of list of str or `None`, default : `None`
            This list specifies what information from the `item_features` to
            put in the `x` vector. Each inner list corresponds to a different
            field. Accepted list values consist of the names of
            features. If `None`, no info is added.
        n_factors : int, default: 10
            Number of latent factors in low-rank approximation.
        dev_ratio : float, default : 0.3
            Ratio of `trainset` to dedicate to development data set to identify
            best model. Should be either positive and smaller than the number
            of samples or a float in the (0, 1) range.
        patience : int, default : 10
            Number of epochs without improvement to wait before ending
            training. The last best model is kept.
        n_epochs : int, default : 50
            Maximum number of training epochs.
        batch_size: int, default : `None`
            Batch size to use for training. If set to `None`, then training is
            done using full dataset.
        init_std : float, default : 0.01
            The standard deviation of the normal distribution for
            initialization.
        lr : float, default : 0.001
            Learning rate for optimization method.
        reg : float, default : 0.02
            Strength of L2 regularization. It can be disabled by setting it to
            zero.
        refit : bool, default : `True`
            Determines whether to refit model with all training data for the
            best number of epochs found with the development data set.
        binary : bool, default : `False`
            Whether to use a sigmoid on the output to obtain a binary output.
        random_state : int, default : `None`
            Determines the RNG that will be used for initialization. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same initialization over multiple calls to
            ``fit()``. If ``None``, the current RNG from torch is used.
        verbose : bool, default : `False`
            Level of verbosity.
    """

    def __init__(self, rating_lst=('userID', 'itemID'), user_lst=None,
                 item_lst=None, n_factors=10, dev_ratio=0.3, patience=10,
                 n_epochs=50, batch_size=None, init_std=0.01, lr=0.001,
                 reg=0.02, refit=True, binary=False, random_state=None,
                 verbose=False, **kwargs):

        warnings.warn('This class is not yet ready.')

        super().__init__(**kwargs)

        self.rating_lst = rating_lst if rating_lst is not None else []
        self.user_lst = user_lst if user_lst is not None else []
        self.item_lst = item_lst if item_lst is not None else []
        self.n_factors = n_factors
        self.dev_ratio = dev_ratio
        self.patience = patience
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_std = init_std
        self.lr = lr
        self.reg = reg
        self.refit = refit
        self.binary = binary
        self.random_state = random_state
        self.verbose = verbose

        self.model = None

    def fit(self, trainset):

        # Set pre-requirements
        torch.set_default_dtype(torch.float64)  # use float64
        super().fit(trainset)
        self.set_random_state()

        # Construct data and initialize model
        self._construct_fit_data()
        self.model = FFMtorchNN(self.field_dims, n_factors=self.n_factors,
                                init_std=self.init_std, binary=self.binary)

        # Train model
        self._fit()

        return self
