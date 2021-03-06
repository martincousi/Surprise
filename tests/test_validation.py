"""
Module for testing the validation module.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from surprise.accuracy import neg_rmse, neg_mae
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import model_selection as ms


def test_cross_validate():

    # First test with a specified CV iterator.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    folds_files = [(current_dir + '/custom_train',
                    current_dir + '/custom_test')]

    reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                    rating_scale=(1, 5))
    data = Dataset.load_from_folds(folds_files=folds_files, reader=reader)

    algo = NormalPredictor()
    pkf = ms.PredefinedKFold()
    ret = ms.cross_validate(algo, data, measures=[['neg_rmse', neg_rmse],
                                                  ['neg_mae', neg_mae]],
                            cv=pkf,
                            verbose=1)
    # Basically just test that keys (dont) exist as they should
    assert len(ret['test_neg_rmse']) == 1
    assert len(ret['test_neg_mae']) == 1
    assert len(ret['fit_time']) == 1
    assert len(ret['test_time']) == 1
    assert 'test_fcp' not in ret
    assert 'train_neg_rmse' not in ret
    assert 'train_neg_mae' not in ret

    # Test that 5 fold CV is used when cv=None
    # Also check that train_* key exist when return_train_measures is True.
    data = Dataset.load_from_file(current_dir + '/custom_dataset', reader)
    ret = ms.cross_validate(algo, data,  measures=[['neg_rmse', neg_rmse],
                                                  ['neg_mae', neg_mae]],
                            cv=None, return_train_measures=True, verbose=True)
    assert len(ret['test_neg_rmse']) == 5
    assert len(ret['test_neg_mae']) == 5
    assert len(ret['fit_time']) == 5
    assert len(ret['test_time']) == 5
    assert len(ret['train_neg_rmse']) == 5
    assert len(ret['train_neg_mae']) == 5
