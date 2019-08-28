"""
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .algo_base import AlgoBase
from .predictions import PredictionImpossible


class BaselineOnly(AlgoBase):
    """Algorithm predicting the baseline estimate for given user and item.

    :math:`\hat{r}_{ui} = b_{ui} = \mu + b_u + b_i`

    If user :math:`u` is unknown, then the bias :math:`b_u` is assumed to be
    zero. The same applies for item :math:`i` with :math:`b_i`.

    See section 2.1 of :cite:`Koren:2010` for details.

    Args:
        bsl_options(dict): A dictionary of options for the baseline estimates
            computation. See :ref:`baseline_estimates_configuration` for
            accepted options.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is False.
    """

    def __init__(self, bsl_options=None, verbose=False):

        if bsl_options is None:
            bsl_options = {}

        AlgoBase.__init__(self, bsl_options=bsl_options)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.bu, self.bi = self.compute_baselines()

        return self

    def estimate(self, u, i, *_):

        knows_user = self.trainset.knows_user(u)
        knows_item = self.trainset.knows_item(i)

        if not (knows_user or knows_item):
            raise PredictionImpossible('Unknown user and item.')

        est = self.trainset.global_mean
        if knows_user:
            est += self.bu[u]
        if knows_item:
            est += self.bi[i]

        return est
