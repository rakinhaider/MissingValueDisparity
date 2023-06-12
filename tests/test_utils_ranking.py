import numpy as np
import pandas as pd

from utils_ranking import conditional_ndcg, ndd
from unittest import TestCase
from sklearn.metrics import ndcg_score


class TestUtilsRanking(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        np.random.seed(47)
        true = pd.DataFrame({0: np.random.rand(12)})
        true['sex'] = np.random.randint(0, 2, 12)
        true['label'] = np.random.randint(0, 2, 12)
        true.set_index(keys=['sex', 'label'], append=True, inplace=True)
        true = true.sort_values(by=[0], ignore_index=False, ascending=False)
        self.true = true[0]
        self.score = pd.Series(np.random.rand(12), index=true.index)

    def test_conditional_ndcg(self):
        output = conditional_ndcg([self.true], [self.score])
        expected = ndcg_score([self.true], [self.score], ignore_ties=True)
        assert np.allclose(expected, output)

    def test_conditional_ndcg_priv(self):
        output = conditional_ndcg([self.true], [self.score], privilege=1)
        expected = 0.32767649949308003
        np.allclose(output, expected)

    def test_conditional_ndcg_unpriv(self):
        output = conditional_ndcg([self.true], [self.score], privilege=0)
        expected = 0.417544712425969
        np.allclose(output, expected)

    def test_conditional_ndcg_sum(self):
        output_p = conditional_ndcg([self.true], [self.score], privilege=1)
        output_u = conditional_ndcg([self.true], [self.score], privilege=0)
        output = conditional_ndcg([self.true], [self.score])
        assert np.allclose(output, output_p + output_u)

        output_p = conditional_ndcg([self.true], [self.score], target=1)
        output_u = conditional_ndcg([self.true], [self.score], target=0)
        output = conditional_ndcg([self.true], [self.score])
        assert np.allclose(output, output_p + output_u)

        output_p = conditional_ndcg([self.true], [self.score],
                                    privilege=1, target=1)
        output_u = conditional_ndcg([self.true], [self.score],
                                    privilege=1, target=0)
        output = conditional_ndcg([self.true], [self.score], privilege=1)
        assert np.allclose(output, output_p + output_u)

    def test_ndd(self):
        np.random.seed(47)
        rank = np.arange(12)
        orig_rank = np.arange(12)
        np.random.shuffle(rank)
        np.random.shuffle(orig_rank)
        df = pd.DataFrame({'rank': rank, 'orig_rank': orig_rank,
                           'sex': np.random.randint(0, 2, 12),
                           })
        df.set_index(keys=['sex'], append=True, inplace=True)
        print(df)
        output = ndd(df['rank'], df['orig_rank'])
        print(output)