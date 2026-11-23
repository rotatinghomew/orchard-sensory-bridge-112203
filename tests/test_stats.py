"""统计函数核对：t 分位数、区间、合并异质性、ANOVA。"""

import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensory.stats import (
    Estimate,
    chi2_survival,
    f_survival,
    inverse_variance_pool,
    mean_ci,
    one_way_anova,
    pearson,
    t_cdf,
    t_quantile,
)


class StatsTest(unittest.TestCase):
    def test_t_cdf_known_points(self):
        # 标准表值：df=8 时 P(T<=2.306)≈0.975；df=1 时 P(T<=0)=0.5
        self.assertAlmostEqual(t_cdf(2.306, 8), 0.975, places=3)
        self.assertAlmostEqual(t_cdf(0.0, 5), 0.5, places=6)
        self.assertAlmostEqual(t_cdf(-2.306, 8), 0.025, places=3)

    def test_t_quantile_inverts_cdf(self):
        for df in (2, 7, 20):
            q = t_quantile(0.95, df)
            self.assertAlmostEqual(t_cdf(q, df), 0.95, places=5)

    def test_mean_ci_symmetry(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        estimate = mean_ci(values, 0.9)
        self.assertAlmostEqual(estimate.mean, 3.0)
        self.assertAlmostEqual(estimate.low + estimate.high, 6.0)
        self.assertLess(estimate.low, estimate.mean)
        self.assertLess(estimate.mean, estimate.high)

    def test_inverse_variance_pool_and_q(self):
        e1 = Estimate(mean=1.0, low=0.9, high=1.1, se=0.05, n=8)
        e2 = Estimate(mean=2.0, low=1.92, high=2.08, se=0.04, n=8)
        pooled = inverse_variance_pool([e1, e2], 0.9)
        self.assertEqual(pooled.session_count, 2)
        self.assertTrue(e1.mean < pooled.mean < e2.mean)
        # 差异相对标准误很大时，Q 应显著。
        self.assertLess(pooled.q_p, 0.05)
        identical = inverse_variance_pool([e1, e1], 0.9)
        self.assertGreater(identical.q_p, 0.99)

    def test_chi2_and_f_survival(self):
        self.assertAlmostEqual(chi2_survival(0.0, 3), 1.0)
        self.assertTrue(0 < chi2_survival(2.0, 3) < 1)
        self.assertAlmostEqual(f_survival(0.0, 2, 10), 1.0)
        self.assertTrue(0 < f_survival(5.0, 2, 10) < 1)

    def test_pearson_and_anova(self):
        self.assertAlmostEqual(pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
        self.assertIsNone(pearson([1, 1, 1], [2, 3, 4]))
        result = one_way_anova([[1, 1.1, 0.9], [5, 5.1, 4.9]])
        f_stat, p_value, _sd, means = result
        self.assertGreater(f_stat, 100)
        self.assertLess(p_value, 0.001)
        self.assertEqual(means, [1.0, 5.0])
        self.assertIsNone(one_way_anova([[1, 2]]))


if __name__ == "__main__":
    unittest.main()
