"""声明层：容许带判定、窗口覆盖、通用/局部声明、印签历史。

判定语义（资料格式.md）：
- 置信区间完全落在容许带内才支持“与原果口感相当”；区间碰到边界是“未定”，
  完全在带外才是“超出容许”；没检出差异绝不等于已证明等同。
- 参照场次数不足、比较网络断开、日龄窗口无证据，一律是证据不足，不是不合格。
- 通用声明要求标签请求里承诺的每个货批 × 每个允许参照 × 整个窗口都被支持；
  否则只给有支撑的局部文字与适用范围，未评到的搭配写成“未评/证据不足”。
- 已印未发的印签按签发时点重放证据；失效参照可迫使其停用，
  仍有独立比较路径撑住的印签继续有效。
"""

from __future__ import annotations

from dataclasses import dataclass

from .stats import PooledEstimate, inverse_variance_pool
from .study import Edge, Study

INSIDE = "inside"          # 区间完全在容许带内：支持等同
CROSSES = "crosses"        # 区间碰到容许边界：未定，不能声称等同
OUTSIDE = "outside"        # 区间完全在带外：超出容许
UNKNOWN = "unknown"        # 证据不足/网络断开：不是不合格

VERDICT_TEXT = {
    INSIDE: "支持等同（区间在容许带内）",
    CROSSES: "未定（区间触及容许边界，不能声称等同）",
    OUTSIDE: "超出容许（区间完全在容许带外）",
    UNKNOWN: "证据不足（不等于不合格）",
}


@dataclass(frozen=True)
class AttributeVerdict:
    attribute: str
    margin: float
    estimate: PooledEstimate | None
    verdict: str
    note: str


@dataclass(frozen=True)
class PairAssessment:
    cargo: str
    anchor: str
    storage_days: int
    reference_sessions: tuple[str, ...]
    session_count_ok: bool
    connected: bool
    edges: tuple[Edge, ...]
    aroma: AttributeVerdict | None
    crisp: AttributeVerdict | None
    drift_warnings: tuple[str, ...]
    exclusion_notes: tuple[str, ...]

    @property
    def supported(self):
        if not (self.connected and self.session_count_ok):
            return False
        return self.aroma.verdict == INSIDE and self.crisp.verdict == INSIDE


@dataclass(frozen=True)
class ItemAssessment:
    item: str
    pairs: tuple[PairAssessment, ...]
    observed_days: tuple[int, ...]
    covered_days: tuple[int, ...]
    window_covered: bool
    status: str
    local_text: str


@dataclass(frozen=True)
class ClaimAssessment:
    request_id: str
    items: tuple[str, ...]
    anchors: tuple[str, ...]
    from_day: int
    to_day: int
    item_assessments: tuple[ItemAssessment, ...]
    general_claim: bool
    general_text: str
    local_texts: tuple[str, ...]
    unsupported_pairings: tuple[str, ...]
    invalidation_notes: tuple[str, ...]
    shared_anchor_notes: tuple[str, ...]


def _classify(estimate: PooledEstimate | None, margin: float, enough_sessions: bool, connected: bool):
    if estimate is None or not (enough_sessions and connected):
        return UNKNOWN
    if estimate.low > margin or estimate.high < -margin:
        return OUTSIDE
    if estimate.low >= -margin and estimate.high <= margin:
        return INSIDE
    return CROSSES


def _window_coverage(observed_days, from_day, to_day):
    """已评日龄只能为实测日或有两端有效实测的中间范围作证，不能外推。"""
    days = sorted(observed_days)
    covered = set()
    for day in range(from_day, to_day + 1):
        if day in days:
            covered.add(day)
            continue
        lower = [d for d in days if d <= day]
        upper = [d for d in days if d >= day]
        if lower and upper and max(lower) >= from_day and min(upper) <= to_day:
            covered.add(day)
    return tuple(sorted(covered))


def assess_pair(study: Study, cargo, anchor, storage_days, as_of, drift_flags):
    """评估单个 (货批, 参照, 日龄) 搭配，含共同参照异常传播提示。"""
    edges = [
        e
        for e in study.build_edges(as_of)
        if e.cargo == cargo and e.anchor == anchor and e.storage_days == storage_days
    ]
    exclusion_notes = tuple(sorted({e.exclusion for e in edges if e.exclusion}))
    eligible = [e for e in edges if e.eligible]
    sessions = tuple(sorted({e.session for e in eligible}))
    min_ref = study.contract["min_reference_sessions"]
    count_ok = len(sessions) >= min_ref
    connected = bool(eligible)

    # 共同影响：同一参照被多个批次共享时，其漂移/失效提示要出现在每个依赖声明上。
    warnings = []
    for row in drift_flags:
        if row.anchor == anchor and row.flagged:
            warnings.append(
                f"共享参照 {anchor} 的 {row.attribute} 在场次 {','.join(row.sessions)} "
                f"出现跨场漂移（ANOVA p={row.p_value}，极差 {row.spread}）；"
                "该异常同时影响所有以它为锚的批次，各份区间不得独立重复使用"
            )

    def attribute_result(attribute, margin):
        if not eligible:
            note = "无有效同场配对边：" + ("；".join(exclusion_notes) or "比较网络断开")
            return AttributeVerdict(attribute, margin, None, UNKNOWN, note)
        estimates = [getattr(e, attribute) for e in eligible]
        pooled = inverse_variance_pool(estimates, study.contract["confidence"])
        verdict = _classify(pooled, margin, count_ok, connected)
        if not count_ok:
            note = (
                f"有效参照场次 {len(sessions)} < 最低 {min_ref}（同场重复码不计不同场），"
                "区间仅供描述，不能支撑等同声明"
            )
        elif verdict == CROSSES:
            note = f"合并区间 {pooled.as_text()} 触及 ±{margin} 容许边界"
        elif verdict == OUTSIDE:
            note = f"合并区间 {pooled.as_text()} 完全超出 ±{margin} 容许带"
        else:
            note = f"合并区间 {pooled.as_text()} 落在 ±{margin} 容许带内"
        if pooled.q_p < 0.1 and count_ok:
            note += f"；场次间异质性 Q 检验 p={pooled.q_p:.2f}，合并需谨慎"
        return AttributeVerdict(attribute, margin, pooled, verdict, note)

    aroma = attribute_result("aroma", study.contract["aroma_margin"])
    crisp = attribute_result("crisp", study.contract["crisp_margin"])
    return PairAssessment(
        cargo=cargo,
        anchor=anchor,
        storage_days=storage_days,
        reference_sessions=sessions,
        session_count_ok=count_ok,
        connected=connected,
        edges=tuple(eligible),
        aroma=aroma,
        crisp=crisp,
        drift_warnings=tuple(warnings),
        exclusion_notes=exclusion_notes,
    )


def assess_request(study: Study, request, as_of, drift_flags):
    """评估一个标签请求在 as_of 时点能写什么。"""
    item_assessments = []
    unsupported = []
    for item in request["items"]:
        pairs = []
        observed = set()
        for anchor in request["anchors"]:
            window_edges = [
                e
                for e in study.build_edges(as_of)
                if e.cargo == item
                and e.anchor == anchor
                and request["from_day"] <= e.storage_days <= request["to_day"]
            ]
            if not window_edges:
                # 从未同场评过的搭配：保留断开记录，但不贡献实测日龄。
                pairs.append(
                    PairAssessment(
                        cargo=item,
                        anchor=anchor,
                        storage_days=request["from_day"],
                        reference_sessions=(),
                        session_count_ok=False,
                        connected=False,
                        edges=(),
                        aroma=AttributeVerdict(
                            "aroma",
                            study.contract["aroma_margin"],
                            None,
                            UNKNOWN,
                            "该货批与参照从未在同场配对，比较网络断开",
                        ),
                        crisp=AttributeVerdict(
                            "crisp",
                            study.contract["crisp_margin"],
                            None,
                            UNKNOWN,
                            "该货批与参照从未在同场配对，比较网络断开",
                        ),
                        drift_warnings=(),
                        exclusion_notes=(),
                    )
                )
                continue
            for day in sorted({e.storage_days for e in window_edges}):
                pair = assess_pair(study, item, anchor, day, as_of, drift_flags)
                pairs.append(pair)
                if pair.connected:
                    observed.add(day)
        covered = _window_coverage(observed, request["from_day"], request["to_day"])
        window_ok = list(covered) == list(range(request["from_day"], request["to_day"] + 1))
        status, local_text = _item_status(item, pairs, covered, request, window_ok)
        if status != "supported":
            unsupported.append(f"{item}: {local_text}")
        item_assessments.append(
            ItemAssessment(
                item=item,
                pairs=tuple(pairs),
                observed_days=tuple(sorted(observed)),
                covered_days=covered,
                window_covered=window_ok,
                status=status,
                local_text=local_text,
            )
        )

    general = all(ia.status == "supported" for ia in item_assessments)
    if general:
        general_text = (
            f"可对 {request['id']} 的全部货批 {', '.join(request['items'])} 在贮藏 "
            f"{request['from_day']}–{request['to_day']} 日窗口使用通用风味/脆度等同声明"
        )
    else:
        general_text = (
            f"{request['id']} 不具备通用声明条件：仅下列有支撑的局部文字可用，"
            "未评到的搭配按证据不足处理，不得写成不合格"
        )
    local_texts = tuple(
        f"{ia.item}：{ia.local_text}" for ia in item_assessments if ia.status != "supported"
    ) + tuple(
        f"{ia.item}：窗口内全部搭配均有支撑"
        for ia in item_assessments if ia.status == "supported"
    )

    invalidation_notes = tuple(
        f"{inv.session} 的参照 {inv.sample} 自 {inv.known} 起失效（{inv.reason}），"
        "只撤除该场次，不影响其他场次的同一参照"
        for inv in study.invalidation_events(as_of)
        if inv.sample in request["anchors"] or any(
            inv.sample in e.anchor for ia in item_assessments for pa in ia.pairs for e in pa.edges
        )
    )
    shared = tuple(
        warning
        for ia in item_assessments
        for pa in ia.pairs
        for warning in pa.drift_warnings
    )
    return ClaimAssessment(
        request_id=request["id"],
        items=tuple(request["items"]),
        anchors=tuple(request["anchors"]),
        from_day=request["from_day"],
        to_day=request["to_day"],
        item_assessments=tuple(item_assessments),
        general_claim=general,
        general_text=general_text,
        local_texts=local_texts,
        unsupported_pairings=tuple(unsupported),
        invalidation_notes=invalidation_notes,
        shared_anchor_notes=tuple(dict.fromkeys(shared)),
    )


def _item_status(item, pairs, covered_days, request, window_ok):
    connected_pairs = [pa for pa in pairs if pa.connected]
    if not connected_pairs:
        return (
            "disconnected",
            "与允许参照之间没有同品种同熟度的有效同场比较，比较网络断开；"
            "不给看似可比的绝对差，按未评处理",
        )
    if not window_ok:
        supported_pairs = [pa for pa in connected_pairs if pa.supported]
        if supported_pairs:
            scope = ", ".join(
                f"{pa.cargo}-{pa.anchor}@{pa.storage_days}日" for pa in supported_pairs
            )
            return (
                "local",
                f"仅 {scope} 有等同支撑；窗口 {request['from_day']}–"
                f"{request['to_day']} 日缺两端实测，不能外推，其余日龄按证据不足处理",
            )
        return (
            "insufficient",
            f"实测日龄 {sorted(covered_days)} 不足以覆盖窗口，且现有搭配未达等同判据，"
            "按证据不足处理（不是不合格）",
        )
    failing = [pa for pa in connected_pairs if not pa.supported]
    if not failing:
        return "supported", "窗口内全部允许搭配的风味与脆度区间均在容许带内"
    reasons = []
    for pa in failing:
        for verdict in (pa.aroma, pa.crisp):
            if verdict.verdict != INSIDE:
                reasons.append(
                    f"{pa.cargo}-{pa.anchor}@{pa.storage_days}日 {verdict.attribute}："
                    f"{VERDICT_TEXT[verdict.verdict]}"
                )
    # local 仅用于“窗口内有部分搭配确实达标、其余缺证”的情形；
    # 全部搭配都未达标（含场次不足/区间触边）归 insufficient，避免暗示已有等同支撑。
    if window_ok and all(
        verdict.verdict in (UNKNOWN, CROSSES)
        for pa in failing
        for verdict in (pa.aroma, pa.crisp)
    ):
        return "insufficient", "；".join(reasons)
    if all(
        verdict.verdict in (UNKNOWN, CROSSES)
        for pa in failing
        for verdict in (pa.aroma, pa.crisp)
    ):
        return "local", "；".join(reasons)
    return "outside", "；".join(reasons)


@dataclass(frozen=True)
class LabelHistory:
    label_id: str
    status: str
    as_of: str
    at_print: ClaimAssessment
    today: ClaimAssessment
    still_valid: bool
    scope_narrowed: bool
    lost_pairings: tuple[tuple, ...]
    shrunk_pairings: tuple[tuple, ...]
    reason: str


def _pair_support_map(assessment: ClaimAssessment):
    """返回 {(货批, 参照, 日龄): 是否等同支撑}，用于比较两个时点的范围变化。"""
    support = {}
    for item in assessment.item_assessments:
        for pair in item.pairs:
            if pair.connected:
                support[(pair.cargo, pair.anchor, pair.storage_days)] = pair.supported
    return support


def assess_label_history(study: Study, label_row, as_of, drift_at_print, drift_now):
    """重放印签签发时点与当前时点的证据，判断停用/缩窄/继续有效。"""
    request = next(r for r in study.label_requests if r["id"] == label_row["id"])
    print_day = label_row["as_of"]
    at_print = assess_request(study, request, print_day, drift_at_print)
    today = assess_request(study, request, as_of, drift_now)

    print_support = _pair_support_map(at_print)
    now_support = _pair_support_map(today)
    lost = sorted(key for key, ok in print_support.items() if ok and not now_support.get(key, False))
    # 即使两边都未达支撑判据，参照场次被失效记录削掉也算范围恶化，需要显式追踪。
    print_sessions = {
        (pa.cargo, pa.anchor, pa.storage_days): pa.reference_sessions
        for item in at_print.item_assessments
        for pa in item.pairs
        if pa.connected
    }
    now_sessions = {
        (pa.cargo, pa.anchor, pa.storage_days): pa.reference_sessions
        for item in today.item_assessments
        for pa in item.pairs
        if pa.connected
    }
    shrunk = sorted(
        key
        for key, sessions in print_sessions.items()
        if len(now_sessions.get(key, ())) < len(sessions)
    )
    narrowed = bool(lost or shrunk)

    if today.general_claim:
        still_valid = True
        reason = "当前仍有独立比较路径撑住全部承诺搭配，印签继续有效"
    elif at_print.general_claim:
        still_valid = False
        reason = (
            f"签发时（{print_day}，失效信息尚未知晓）具备通用支撑；"
            "新发现的失效参照/漂移使部分搭配失去支撑，应停止沿用该印签的通用表述，"
            "仅保留仍有独立路径的局部文字"
        )
    elif shrunk:
        still_valid = False
        reason = (
            f"签发时即未达到通用声明条件（印签不是新证据）；此后失效记录又削掉了部分参照场次："
            f"{', '.join('-'.join(map(str, key)) for key in shrunk)}，适用范围进一步缩小，"
            "当前连描述性区间也只剩单场支撑"
        )
    else:
        still_valid = False
        reason = (
            "签发时即未达到通用声明条件，印签只能作为历史记录；"
            "当前证据状态未再恶化，但也没有新的独立路径使其恢复"
        )
    return LabelHistory(
        label_id=label_row["id"],
        status=label_row["status"],
        as_of=print_day,
        at_print=at_print,
        today=today,
        still_valid=still_valid,
        scope_narrowed=narrowed,
        lost_pairings=tuple(lost),
        shrunk_pairings=tuple(shrunk),
        reason=reason,
    )
