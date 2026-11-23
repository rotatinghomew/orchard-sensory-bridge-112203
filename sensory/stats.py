"""连续分布与估计函数，仅使用 Python 标准库。

约定：
- 置信区间统一用配对差值在评委间的 t 区间（同场差值按同一评委配对）。
- 多场次合并采用逆方差固定效应合并，Q 统计量检查场次间异质性。
- 单因素 ANOVA 的 F 检验用于锚点跨场漂移诊断。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("空序列不能求均值")
    return sum(values) / len(values)


def sample_sd(values):
    values = list(values)
    if len(values) < 2:
        raise ValueError("至少两个观测才能求样本标准差")
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


# ---------------------------------------------------------------------------
# 正则化不完全 Beta 函数（Numerical Recipes 形式），用于 t 与 F 分布。
# ---------------------------------------------------------------------------
def _beta_continued_fraction(a, b, x):
    max_iter = 200
    epsilon = 3e-12
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        term = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + term * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + term / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        term = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + term * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + term / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def regularized_beta(a, b, x):
    """I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    front = math.exp(log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def t_cdf(t, df):
    """学生 t 分布的累积分布函数。"""
    x = df / (df + t * t)
    incomplete = regularized_beta(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * incomplete if t > 0 else 0.5 * incomplete


def t_quantile(probability, df):
    """二分法求 t 分位数；df 为正整数。"""
    if df <= 0:
        raise ValueError("自由度必须为正")
    low, high = 0.0, 100.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if t_cdf(mid, df) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def chi2_survival(stat, df):
    """卡方生存函数 P(X>stat)，用正则化 Gamma（P(a,x)=regularized_beta 形式）。"""
    if df <= 0:
        raise ValueError("自由度必须为正")
    if stat <= 0:
        return 1.0
    x = df / (df + stat)
    # P(chi2 > stat) = regularized_gamma_lower(df/2, stat/2) 的补 = I_x(df/2, 1/2)
    return regularized_beta(df / 2.0, 0.5, x)


def f_survival(f, df1, df2):
    """F 分布生存函数。"""
    if f <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f)
    # P(F>f) = I_{df2/(df2+df1 f)}(df2/2, df1/2)
    return regularized_beta(df2 / 2.0, df1 / 2.0, x)


@dataclass(frozen=True)
class Estimate:
    """均值及其置信区间。"""

    mean: float
    low: float
    high: float
    se: float
    n: int
    sd: float | None = None
    confidence: float = 0.9

    def as_text(self, digits=2):
        return f"{self.mean:+.{digits}f} [{self.low:+.{digits}f}, {self.high:+.{digits}f}]"


def mean_ci(values, confidence=0.9):
    """普通均值的 t 置信区间。"""
    values = list(values)
    n = len(values)
    if n < 2:
        raise ValueError("置信区间至少需要两个观测")
    m = mean(values)
    sd = sample_sd(values)
    se = sd / math.sqrt(n)
    half = t_quantile(1 - (1 - confidence) / 2, n - 1) * se
    return Estimate(mean=m, low=m - half, high=m + half, se=se, n=n, sd=sd, confidence=confidence)


@dataclass(frozen=True)
class PooledEstimate:
    """逆方差固定效应合并结果。"""

    mean: float
    low: float
    high: float
    se: float
    confidence: float
    q: float
    q_df: int
    q_p: float
    weights: tuple[float, ...]
    session_count: int

    def as_text(self, digits=2):
        return f"{self.mean:+.{digits}f} [{self.low:+.{digits}f}, {self.high:+.{digits}f}]"


def inverse_variance_pool(estimates, confidence=0.9):
    """对若干场次的 Estimate 做固定效应合并，附 Cochran Q 异质性检验。"""
    estimates = list(estimates)
    if not estimates:
        raise ValueError("没有可合并的估计")
    if any(e.se <= 0 for e in estimates):
        raise ValueError("待合并估计存在零标准误，无法按逆方差合并（评委间需要有变异）")
    weights = [1.0 / (e.se ** 2) for e in estimates]
    total_w = sum(weights)
    pooled_mean = sum(w * e.mean for w, e in zip(weights, estimates)) / total_w
    se = math.sqrt(1.0 / total_w)
    q = sum(w * (e.mean - pooled_mean) ** 2 for w, e in zip(weights, estimates))
    q_df = len(estimates) - 1
    q_p = chi2_survival(q, q_df) if q_df > 0 else 1.0
    total_df = sum(e.n - 1 for e in estimates)
    half = t_quantile(1 - (1 - confidence) / 2, max(total_df, 1)) * se
    return PooledEstimate(
        mean=pooled_mean,
        low=pooled_mean - half,
        high=pooled_mean + half,
        se=se,
        confidence=confidence,
        q=q,
        q_df=q_df,
        q_p=q_p,
        weights=tuple(weights),
        session_count=len(estimates),
    )


def pearson(xs, ys):
    """Pearson 相关；方差为 0 时返回 None。"""
    xs, ys = list(xs), list(ys)
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("相关需要两个等长且长度至少为 2 的序列")
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def one_way_anova(groups):
    """单因素 ANOVA，返回 (F, 生存p, 组内SD合并, 组均值)；不足两组返回 None。"""
    groups = [list(g) for g in groups if g]
    k = len(groups)
    if k < 2 or any(len(g) < 1 for g in groups):
        return None
    all_values = [v for g in groups for v in g]
    grand = mean(all_values)
    ss_between = sum(len(g) * (mean(g) - grand) ** 2 for g in groups)
    ss_within = sum(sum((v - mean(g)) ** 2 for v in g) for g in groups)
    df_between = k - 1
    df_within = len(all_values) - k
    if df_within <= 0 or ss_within <= 0:
        return None
    f_stat = (ss_between / df_between) / (ss_within / df_within)
    p_value = f_survival(f_stat, df_between, df_within)
    pooled_sd = math.sqrt(ss_within / df_within)
    return f_stat, p_value, pooled_sd, [mean(g) for g in groups]
