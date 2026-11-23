"""冷藏苹果盲评的可反复更新感官比较。

模块划分：
- stats：仅依赖标准库的连续分布与估计函数（t 分位数、卡方生存、相关、合并效应）。
- study：资料装配、盲码解盲、失效剔除、同条件比较边与可比网络组件。
- diagnostics：同批重复样、评委跨场重复评分、锚点漂移/量表漂移、上样顺序效应。
- claims：按容许带给出差异与置信区间、声明窗口覆盖、印签历史与范围收缩。
- planning：候选补评场次的留样可行性与可恢复组合。
- report：面向运营的文字报告装配。

所有结论只在“同品种 + 同熟度 + 同场配对”的有效边上成立；断开的网络不给绝对差。
"""

from .study import Study, Edge, JoinedScore
from .claims import (
    ClaimAssessment,
    ItemAssessment,
    PairAssessment,
    AttributeVerdict,
    assess_request,
    assess_label_history,
)
from .planning import CandidatePlan, plan_candidates
from .diagnostics import (
    CategoryMismatchRow,
    anchor_drift_table,
    category_mismatch_table,
    order_effect_table,
    repeatability_table,
    rater_repeat_table,
)

__all__ = [
    "Study",
    "Edge",
    "JoinedScore",
    "ClaimAssessment",
    "ItemAssessment",
    "PairAssessment",
    "AttributeVerdict",
    "assess_request",
    "assess_label_history",
    "CandidatePlan",
    "plan_candidates",
    "repeatability_table",
    "rater_repeat_table",
    "anchor_drift_table",
    "category_mismatch_table",
    "order_effect_table",
]
