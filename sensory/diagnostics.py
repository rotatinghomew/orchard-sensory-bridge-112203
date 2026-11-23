"""诊断层：重复样可靠性、评委跨场重复评分、锚点/量表漂移、上样顺序效应。

这些诊断不直接产生“等同”结论，而是解释比较边的不确定性来源：
- 同批重复样相关低，说明评委对同一实体样品的重复判分噪声大，差值区间会更宽。
- 同一评委跨场给同一实体锚的分跳动，是量表漂移的个体证据。
- 锚点跨场均值差异经 ANOVA 标记；被多个批次共享的锚，异常沿共同影响传播。
"""

from __future__ import annotations

from dataclasses import dataclass

from .stats import mean, one_way_anova, pearson, sample_sd
from .study import Study


@dataclass(frozen=True)
class RepeatabilityRow:
    session: str
    sample: str
    attribute: str
    raters: int
    pearson_r: float | None
    repeat_sd: float
    mean_signed_diff: float


@dataclass(frozen=True)
class RaterRepeatRow:
    rater: str
    sample: str
    appearances: tuple[str, ...]
    aroma_sd: float
    crisp_sd: float


@dataclass(frozen=True)
class AnchorDriftRow:
    anchor: str
    attribute: str
    sessions: tuple[str, ...]
    means: tuple[float, ...]
    spread: float
    f_stat: float | None
    p_value: float | None
    flagged: bool


@dataclass(frozen=True)
class OrderEffectRow:
    attribute: str
    rater_sessions: int
    mean_signed_corr: float
    mean_abs_corr: float
    flagged: bool


@dataclass(frozen=True)
class CategoryMismatchRow:
    session: str
    cargo: str
    anchor: str
    cargo_variety: str
    anchor_variety: str
    cargo_maturity: str
    anchor_maturity: str
    reason: str


DRIFT_P = 0.10
ORDER_FLAG = 0.20


def _score_value(score, attribute_index):
    return score.aroma if attribute_index == 0 else score.crisp


def repeatability_table(study: Study):
    """同场重复码：同一评委对同一实体样品两次判分的相关与重复标准差。"""
    rows = []
    for session, sample, codes, code_scores in study.repeatability_records():
        common_raters = set(code_scores[0])
        for extra in code_scores[1:]:
            common_raters &= set(extra)
        for attribute_index, attribute in ((0, "aroma"), (1, "crisp")):
            first, second = [], []
            for rater in sorted(common_raters):
                first.append(_score_value(code_scores[0][rater], attribute_index))
                second.append(_score_value(code_scores[1][rater], attribute_index))
            if len(first) < 2:
                continue
            corr = pearson(first, second)
            diffs = [a - b for a, b in zip(first, second)]
            # 重复测量的单次标准差：sd(差值)/sqrt(2)
            repeat_sd = sample_sd(diffs) / (2 ** 0.5)
            rows.append(
                RepeatabilityRow(
                    session=session,
                    sample=sample,
                    attribute=attribute,
                    raters=len(first),
                    pearson_r=corr,
                    repeat_sd=repeat_sd,
                    mean_signed_diff=mean(diffs),
                )
            )
    return tuple(rows)


def rater_repeat_table(study: Study):
    """同一评委在多个场次给同一实体样品的评分波动（评委个人量表稳定性）。"""
    rows = []
    for rater, by_sample in study.rater_cross_session_scores().items():
        for sample, appearances in sorted(by_sample.items()):
            if len(appearances) < 2:
                continue
            sessions = tuple(a[0] for a in appearances)
            aromas = [a[1] for a in appearances]
            crisps = [a[2] for a in appearances]
            rows.append(
                RaterRepeatRow(
                    rater=rater,
                    sample=sample,
                    appearances=sessions,
                    aroma_sd=sample_sd(aromas),
                    crisp_sd=sample_sd(crisps),
                )
            )
    return tuple(sorted(rows, key=lambda r: (r.sample, r.rater)))


def anchor_drift_table(study: Study, as_of, p_threshold=DRIFT_P):
    """锚点跨场单因素 ANOVA；同实体锚的跨场均值差异显著即标记漂移。

    注意：原始均值跨场不可直接当作货批差异，这里只用于判断量表/锚稳定性，
    并通过 Study.anchor_shared_impact 把标记传播到所有依赖该锚的比较。
    """
    presence = study.anchor_presence(as_of)
    rows = []
    for anchor, sessions in presence.items():
        if len(sessions) < 2:
            continue
        for attribute_index, attribute in ((0, "aroma"), (1, "crisp")):
            groups = []
            for session in sessions:
                rater_values = study.rater_sample_scores(session, anchor)
                groups.append([v[attribute_index] for v in rater_values.values()])
            anova = one_way_anova(groups)
            means = tuple(mean(g) for g in groups)
            spread = max(means) - min(means)
            if anova is None:
                f_stat = p_value = None
                flagged = False
            else:
                f_stat, p_value, _pooled_sd, _ = anova
                flagged = p_value < p_threshold
            rows.append(
                AnchorDriftRow(
                    anchor=anchor,
                    attribute=attribute,
                    sessions=sessions,
                    means=tuple(round(m, 3) for m in means),
                    spread=round(spread, 3),
                    f_stat=None if f_stat is None else round(f_stat, 3),
                    p_value=None if p_value is None else round(p_value, 3),
                    flagged=flagged,
                )
            )
    return tuple(rows)


def category_mismatch_table(study: Study, as_of):
    """列出同场但品种或熟度不等价的货批—锚配对：刻度不能架在这些样品上。"""
    rows = []
    for edge in study.build_edges(as_of):
        if edge.eligible:
            continue
        cargo_info = study.samples[edge.cargo]
        anchor_info = study.samples[edge.anchor]
        if not (edge.same_variety and edge.same_maturity):
            if not edge.same_variety:
                reason = "品种不同"
            else:
                reason = "同品种但熟度不同"
            rows.append(
                CategoryMismatchRow(
                    session=edge.session,
                    cargo=edge.cargo,
                    anchor=edge.anchor,
                    cargo_variety=cargo_info.variety,
                    anchor_variety=anchor_info.variety,
                    cargo_maturity=cargo_info.maturity,
                    anchor_maturity=anchor_info.maturity,
                    reason=reason,
                )
            )
    return tuple(rows)


def order_effect_table(study: Study):
    """每位评委每场内部，上样顺序与评分的相关；均值相关显著偏向一侧才判系统顺序效应。"""
    per_session_rater: dict[tuple, dict[str, object]] = {}
    for score in study.joined:
        per_session_rater.setdefault((score.session, score.rater), []).append(score)
    rows = []
    for attribute_index, attribute in ((0, "aroma"), (1, "crisp")):
        correlations = []
        for (_session, _rater), scores in per_session_rater.items():
            if len(scores) < 4:
                continue
            orders = [s.serving_order for s in scores]
            values = [
                s.aroma if attribute == "aroma" else s.crisp for s in scores
            ]
            corr = pearson(orders, values)
            if corr is not None:
                correlations.append(corr)
        mean_signed = mean(correlations)
        mean_abs = mean([abs(c) for c in correlations])
        rows.append(
            OrderEffectRow(
                attribute=attribute,
                rater_sessions=len(correlations),
                mean_signed_corr=round(mean_signed, 3),
                mean_abs_corr=round(mean_abs, 3),
                flagged=abs(mean_signed) >= ORDER_FLAG,
            )
        )
    return tuple(rows)
