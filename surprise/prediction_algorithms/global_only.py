"""
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .algo_base import AlgoBase


class GlobalOnly(AlgoBase):
    """Algorithm providing the global mean as prediction.
    """

    def __init__(self):

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        return self

    def estimate(self, *_):
        """ `AlgoBase.default_prediction` is equivalent to `GlobalOnly`.
        """

        return self.default_prediction()
