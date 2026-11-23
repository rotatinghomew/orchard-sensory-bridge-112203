"""补评场次规划：留样可行性与可恢复的组合。

规则（资料格式.md）：
- 同场每个列入的参照至少耗用 sessions.servings 份参照果留样。
- keep_until 当日可用，次日不可用；货批留样由组织者自备，不占参照果留样。
- 同一份已耗用的留样不能事后补给多个场次；因此候选场次要成套检查冲突。
- 新场次必须同时连接需要恢复可比的组合（货批与允许锚同场出现才有比较边）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .study import Study, _parse_day


@dataclass(frozen=True)
class AnchorNeed:
    anchor: str
    servings_required: int
    reserve_available: int | None
    keep_until: str | None
    feasible: bool
    reason: str


@dataclass(frozen=True)
class CandidatePlan:
    session_id: str
    day: str
    servings_per_anchor: int
    candidates: tuple[str, ...]
    anchor_needs: tuple[AnchorNeed, ...]
    feasible_alone: bool
    restorable_pairings: tuple[str, ...]
    disallowed_links: tuple[str, ...]
    unconnected_candidates: tuple[str, ...]
    note: str


def _reserve_index(study: Study):
    return {row["sample"]: row for row in study.reserves}


def evaluate_candidate(study: Study, session_row):
    reserves = _reserve_index(study)
    needs = []
    feasible = True
    for sample in session_row["candidate"]:
        if study.samples[sample].type != "anchor":
            continue
        required = session_row["servings"]
        reserve = reserves.get(sample)
        if reserve is None:
            ok = False
            reason = "资料中没有该参照的留样记录"
        else:
            keep = _parse_day(reserve["keep_until"])
            session_day = _parse_day(session_row["day"])
            if session_day > keep:
                ok = False
                reason = f"留样 {sample} 仅可用至 {reserve['keep_until']}，场次在其之后"
            elif reserve["servings"] < required:
                ok = False
                reason = (
                    f"留样 {sample} 仅剩 {reserve['servings']} 份，"
                    f"单场至少需要 {required} 份"
                )
            else:
                ok = True
                reason = f"占用 {required} 份后余 {reserve['servings'] - required} 份"
        needs.append(
            AnchorNeed(
                anchor=sample,
                servings_required=required,
                reserve_available=None if reserve is None else reserve["servings"],
                keep_until=None if reserve is None else reserve["keep_until"],
                feasible=ok,
                reason=reason,
            )
        )
        feasible = feasible and ok

    restorable, disallowed, unconnected = _restorable_pairings(study, session_row)
    if feasible:
        note = "单看本场留样可行；能否与其他候选同排还要做组合冲突检查"
    else:
        note = "本场留样不可行，不能排入补评计划"
    return CandidatePlan(
        session_id=session_row["id"],
        day=session_row["day"],
        servings_per_anchor=session_row["servings"],
        candidates=tuple(session_row["candidate"]),
        anchor_needs=tuple(needs),
        feasible_alone=feasible,
        restorable_pairings=tuple(restorable),
        disallowed_links=tuple(disallowed),
        unconnected_candidates=tuple(unconnected),
        note=note,
    )


def _restorable_pairings(study: Study, session_row):
    """候选场内同品种同熟度的货批—锚搭配，区分请求允许与未允许。"""
    requests = {r["id"]: r for r in study.label_requests}
    restorable, disallowed, unconnected = [], [], []
    candidates = session_row["candidate"]
    for cargo in candidates:
        if study.samples[cargo].type != "cargo":
            continue
        cargo_links = []
        for anchor in candidates:
            if study.samples[anchor].type != "anchor":
                continue
            same = (
                study.samples[cargo].variety == study.samples[anchor].variety
                and study.samples[cargo].maturity == study.samples[anchor].maturity
            )
            if not same:
                continue
            cargo_links.append(anchor)
            allowed_in = [
                request_id
                for request_id, request in requests.items()
                if cargo in request["items"] and anchor in request["anchors"]
            ]
            if allowed_in:
                for request_id in allowed_in:
                    request = requests[request_id]
                    restorable.append(
                        f"{request_id}: {cargo}-{anchor}（窗口 "
                        f"{request['from_day']}–{request['to_day']} 日）"
                    )
            else:
                disallowed.append(
                    f"{cargo}-{anchor}：同品种同熟度可连接，但该锚不在任何相关请求的允许参照内"
                )
        if not cargo_links:
            unconnected.append(cargo)
    return sorted(set(restorable)), sorted(set(disallowed)), unconnected


def plan_candidates(study: Study, as_of):
    """返回 (各候选评估, 可同排组合, 冲突说明)。"""
    as_of = _parse_day(as_of) if isinstance(as_of, str) else as_of
    plans = [evaluate_candidate(study, row) for row in study.session_candidates]
    reserves = _reserve_index(study)

    combinations, conflicts = [], []
    feasible_plans = [p for p in plans if p.feasible_alone]
    for i, left in enumerate(feasible_plans):
        for right in feasible_plans[i + 1 :]:
            shared = sorted(
                {n.anchor for n in left.anchor_needs}
                & {n.anchor for n in right.anchor_needs}
            )
            if not shared:
                combinations.append((left.session_id, right.session_id, ()))
                continue
            blocked = []
            for anchor in shared:
                reserve = reserves[anchor]
                demand = left.servings_per_anchor + right.servings_per_anchor
                if reserve["servings"] < demand:
                    blocked.append(
                        f"{anchor} 两场共需 {demand} 份，仅余 {reserve['servings']} 份；"
                        "同一份留样不能事后补两个场次"
                    )
            if blocked:
                conflicts.append((left.session_id, right.session_id, tuple(blocked)))
            else:
                combinations.append((left.session_id, right.session_id, tuple(shared)))

    # 恢复覆盖：哪些当前缺证的请求能被可行候选连到。
    coverage = {}
    for plan in feasible_plans:
        for pairing in plan.restorable_pairings:
            request_id = pairing.split(":", 1)[0]
            coverage.setdefault(request_id, set()).add(
                f"{plan.session_id} -> {pairing}"
            )
    return plans, combinations, conflicts, coverage
