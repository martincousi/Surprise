"""
The :mod:`surprise.accuracy` module provides with tools for computing accuracy
metrics on a set of predictions. We assume higher is always better as in
sklearn and hence we sometime return the opposite of a metric.

Available accuracy metrics:

.. autosummary::
    :nosignatures:

    neg_rmse
    neg_mae
    fcp
    ndcg
    f1
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import math
import warnings

import numpy as np
from six import iteritems


def neg_rmse(predictions, verbose=False):
    """Compute the negative RMSE (Root Mean Squared Error).

    .. math::
        \\text{RMSE} = - \\sqrt{\\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2}.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``False``.

    Returns:
        The negative Root Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    neg_rmse_ = -np.sqrt(mse)

    if verbose:
        print('Negative RMSE: {0:1.4f}'.format(neg_rmse_))

    return neg_rmse_


def neg_mae(predictions, verbose=False):
    """Compute the negative MAE (Mean Absolute Error).

    .. math::
        \\text{MAE} = - \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}|r_{ui} - \\hat{r}_{ui}|

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The negative Mean Absolute Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    neg_mae_ = - np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('Negative MAE:  {0:1.4f}'.format(neg_mae_))

    return neg_mae_


def neg_bce(predictions, verbose=False):
    """Compute the negative BCE (binary cross-entropy).

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.

    Returns:
        The negative binary cross-entropy of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
        ValueError: When true ratings are not binary.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    neg_bce_ = 0.
    for _, _, true_r, est, _ in predictions:
        if true_r == 0.:
            neg_bce_ -= math.log(1. - est)
        elif true_r == 1.:
            neg_bce_ -= math.log(est)
        else:
            raise ValueError('True ratings are not binary.')
    neg_bce_ /= -len(predictions)

    if verbose:
        print('Negative BCE:  {0:1.4f}'.format(neg_bce_))

    return neg_bce_


def fcp(predictions, verbose=False):
    """Compute FCP (Fraction of Concordant Pairs).

    Computed as described in paper `Collaborative Filtering on Ordinal User
    Feedback <http://www.ijcai.org/Proceedings/13/Papers/449.pdf>`_ by Koren
    and Sill, section 5.2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``False``.

    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in iteritems(predictions_u):
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                elif esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = sum(nc_u.values())
    nd = sum(nd_u.values())

    if nc + nd == 0:  # no pairs with different ratings
        fcp_ = 0.
    else:
        fcp_ = nc / (nc + nd)

    if verbose:
        print('FCP:  {0:1.4f}'.format(fcp_))

    return fcp_


def ndcg(predictions, relevance=None, reverse=True, verbose=False):
    """ Compute tie-aware NDCG (Normalized Discounted Cumulative Gain).

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        relevance: Function that takes as input a `Prediction` namedtuple and
            returns its true relevance; a higher relevance means the item
            should have a lower ranked position, and relevance must always be
            non-negative. Default function returns the true rating as the
            relevance.
        reverse: If True, a higher score is preferable. Default is ``True``.
        verbose: If True, will print computed value. Default is ``False``.

    Returns:
        The Normalized Discounted Cumulative Gain.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    def default_relevance(pred):
        return pred.r_ui

    if relevance is None:
        relevance = default_relevance

    def gain(rel):
        return (2 ** rel) - 1

    def discount(gain, rank):
        return gain / math.log2(rank + 2.)

    predictions_u = defaultdict(list)
    for pred in predictions:
        predictions_u[pred.uid].append(pred)

    ndcg_ = 0.
    for uid, preds in iteritems(predictions_u):
        # Compute ideal DCG
        idcg_u = 0.
        # decreasing order of relevance
        true_ranks = sorted(preds, key=lambda pred: relevance(pred),
                            reverse=True)
        for i, pred in enumerate(true_ranks):
            idcg_u += discount(gain(relevance(pred)), i)
        if idcg_u == 0.:
            continue

        # Compute DCG
        dcg_u = 0.
        # decreasing order of estimated rating, unless non-default reverse
        est_ranks = sorted(preds, key=lambda pred: pred.est,
                           reverse=reverse)
        a = 0
        while a < len(est_ranks):
            b = a + 1
            while ((b < len(est_ranks)) and
                    (est_ranks[a].est == est_ranks[b].est)):
                b = b + 1
            avg_gain = 0.
            for pred in est_ranks[a:b]:
                avg_gain += gain(relevance(pred))
            avg_gain /= (b - a)
            for i in range(a, b):
                dcg_u += discount(avg_gain, i)
            a = b
            
        # Compute NDCG
        ndcg_ += dcg_u / idcg_u

    ndcg_ /= len(predictions_u)

    if verbose:
        print('NDCG:  {0:1.4f}'.format(ndcg_))

    return ndcg_


def f1(predictions, threshold, reverse=True, verbose=False):
    """ Compute F1 score.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        threshold (float): Threshold for the rating score to differentiate
            the good items from the bad ones. If a rating is equal to the
            threshold, this item is considered good.
        reverse: If True, a higher score is preferable. Default is ``True``.
        verbose: If True, will print computed value. Default is ``False``.

    Returns:
        The Normalized Discounted Cumulative Gain.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    ratings = np.array([(pred.r_ui, pred.est) for pred in predictions])

    # targets indicate good items with 1
    if reverse:
        targets = ratings >= threshold
    else:
        targets = ratings <= threshold

    true_targets = targets[:, 0]
    est_targets = targets[:, 1]

    tp = np.sum(true_targets & est_targets)
    fp = np.sum(~true_targets & est_targets)
    fn = np.sum(true_targets & ~est_targets)

    if tp == 0:
        f1_ = 0.  # as for sklearn.metrics.f1_score
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_ = 2 * precision * recall / (precision + recall)

    if verbose:
        print('F1:  {0:1.4f}'.format(f1_))

    return f1_
