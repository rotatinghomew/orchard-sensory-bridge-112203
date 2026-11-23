"""资料装配、解盲、失效剔除、比较边与可比网络。

核心规则（见 资料格式.md）：
- blind_map 的 (session, code) 与 scores 对应；同一实体样品可在同场以多码重复出现。
- 跨场锚必须是同实体样品且同熟度；只同品种不同熟度（如 KX 对 Ruby 货批）不是锚。
- invalidations 只让指定场次的指定样品退出，不撤除其他场次的同一参照；
  仅在 known 当日及以后影响新判断，因此所有判定都带 as_of 日期。
- 量表只在同场内、按同一评委配对才可比；跨场的原始分高低不直接比较。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .stats import Estimate, mean, mean_ci, sample_sd


def _parse_day(text):
    return date.fromisoformat(text)


@dataclass(frozen=True)
class Sample:
    sample: str
    variety: str
    maturity: str
    type: str


@dataclass(frozen=True)
class BlindEntry:
    session: str
    code: str
    sample: str
    reveal: date
    storage_days: int


@dataclass(frozen=True)
class Invalidation:
    session: str
    sample: str
    reason: str
    known: date


@dataclass(frozen=True)
class JoinedScore:
    """解盲后的单条评委评分（保留盲码与上样顺序）。"""

    session: str
    code: str
    sample: str
    rater: str
    aroma: float
    crisp: float
    serving_order: int
    storage_days: int


@dataclass(frozen=True)
class Edge:
    """一条可判定的同场配对比较（货批 - 参照）。

    eligible=False 时记录排除原因（失效、品种/熟度不等价、评委不足），
    这类边保留在资料里用于解释为什么不能声明，而不是悄悄丢弃。
    """

    session: str
    cargo: str
    anchor: str
    storage_days: int
    same_variety: bool
    same_maturity: bool
    valid_panel: int
    eligible: bool
    exclusion: str | None
    aroma: Estimate | None = None
    crisp: Estimate | None = None
    rater_pairs: tuple[str, ...] = ()

    @property
    def key(self):
        return self.cargo, self.anchor, self.storage_days


class Study:
    """一份可反复更新的盲评资料；所有判定方法都接受 as_of。"""

    def __init__(self, packet, today):
        self.today = _parse_day(today) if isinstance(today, str) else today
        self.contract = dict(packet["contract"])
        self.samples = {
            row["sample"]: Sample(row["sample"], row["variety"], row["maturity"], row["type"])
            for row in packet["samples"]
        }
        self.blind = [
            BlindEntry(
                row["session"],
                row["code"],
                row["sample"],
                _parse_day(row["reveal"]),
                row["storage_days"],
            )
            for row in packet["blind_map"]
        ]
        self._code_to_entry = {(b.session, b.code): b for b in self.blind}
        self.invalidations = [
            Invalidation(
                row["session"], row["sample"], row["reason"], _parse_day(row["known"])
            )
            for row in packet["invalidations"]
        ]
        self.procedures = {row["id"]: row for row in packet["procedures"]}
        self.label_requests = packet["label_requests"]
        self.labels = packet["labels"]
        self.reserves = packet["reserves"]
        self.session_candidates = packet["sessions"]
        self.joined = self._join_scores(packet["scores"])
        self.sessions = sorted({b.session for b in self.blind})
        self._session_day = {
            sid: _parse_day(self.procedures[sid]["day"]) for sid in self.sessions
        }

    # ------------------------------------------------------------------
    def _join_scores(self, scores):
        joined = []
        for row in scores:
            entry = self._code_to_entry[(row["session"], row["code"])]
            joined.append(
                JoinedScore(
                    session=row["session"],
                    code=row["code"],
                    sample=entry.sample,
                    rater=row["rater"],
                    aroma=float(row["aroma"]),
                    crisp=float(row["crisp"]),
                    serving_order=row["serving_order"],
                    storage_days=entry.storage_days,
                )
            )
        return tuple(joined)

    # ------------------------------------------------------------------
    @staticmethod
    def _as_date(as_of):
        return _parse_day(as_of) if isinstance(as_of, str) else as_of

    def is_invalid(self, session, sample, as_of):
        """该 (场次,样品) 在 as_of 当日是否已退出。"""
        as_of = self._as_date(as_of)
        for inv in self.invalidations:
            if (
                inv.session == session
                and inv.sample == sample
                and inv.known <= as_of
            ):
                return inv
        return None

    def invalidation_events(self, as_of):
        as_of = self._as_date(as_of)
        return [inv for inv in self.invalidations if inv.known <= as_of]

    # ------------------------------------------------------------------
    def session_present_samples(self, session):
        return sorted({b.sample for b in self.blind if b.session == session})

    def storage_days(self, session, sample):
        for b in self.blind:
            if b.session == session and b.sample == sample:
                return b.storage_days
        raise KeyError((session, sample))

    def duplicate_codes(self, session, sample):
        return sorted(
            b.code for b in self.blind if b.session == session and b.sample == sample
        )

    def rater_sample_scores(self, session, sample):
        """同场同样品按评委合并重复码后的 (aroma, crisp, 重复码数)。"""
        codes = self.duplicate_codes(session, sample)
        per_rater: dict[str, list[JoinedScore]] = {}
        for score in self.joined:
            if score.session == session and score.code in codes:
                per_rater.setdefault(score.rater, []).append(score)
        result = {}
        for rater, rows in per_rater.items():
            result[rater] = (
                mean([r.aroma for r in rows]),
                mean([r.crisp for r in rows]),
                len(rows),
            )
        return result

    def raw_rater_code_scores(self, session, code):
        return {
            r.rater: r
            for r in self.joined
            if r.session == session and r.code == code
        }

    # ------------------------------------------------------------------
    def build_edges(self, as_of):
        """构造同场全部货批—参照配对边，并标记资格与排除原因。"""
        edges = []
        min_panel = self.contract["min_valid_panel"]
        for session in self.sessions:
            present = self.session_present_samples(session)
            anchors = [s for s in present if self.samples[s].type == "anchor"]
            cargos = [s for s in present if self.samples[s].type == "cargo"]
            for cargo in cargos:
                for anchor in anchors:
                    edges.append(self._build_edge(session, cargo, anchor, as_of, min_panel))
        return tuple(edges)

    def _build_edge(self, session, cargo, anchor, as_of, min_panel):
        cargo_inv = self.is_invalid(session, cargo, as_of)
        anchor_inv = self.is_invalid(session, anchor, as_of)
        same_variety = self.samples[cargo].variety == self.samples[anchor].variety
        same_maturity = self.samples[cargo].maturity == self.samples[anchor].maturity
        cargo_raters = self.rater_sample_scores(session, cargo)
        anchor_raters = self.rater_sample_scores(session, anchor)
        common = sorted(set(cargo_raters) & set(anchor_raters))
        storage_days = self.storage_days(session, cargo)

        exclusion = None
        if anchor_inv is not None:
            exclusion = f"参照 {anchor} 于 {anchor_inv.known} 失效（{anchor_inv.reason}）"
        elif cargo_inv is not None:
            exclusion = f"货批 {cargo} 于 {cargo_inv.known} 失效（{cargo_inv.reason}）"
        elif not same_variety:
            exclusion = (
                f"品种不等价（{self.samples[cargo].variety} vs "
                f"{self.samples[anchor].variety}），同品种不同熟度的对照不能作锚"
            )
        elif not same_maturity:
            exclusion = (
                f"熟度不等价（{self.samples[cargo].maturity} vs "
                f"{self.samples[anchor].maturity}）"
            )
        elif len(common) < min_panel:
            exclusion = f"有效评委 {len(common)} 人 < 最低 {min_panel} 人"

        aroma = crisp = None
        if exclusion is None:
            aroma_diffs = [
                cargo_raters[r][0] - anchor_raters[r][0] for r in common
            ]
            crisp_diffs = [
                cargo_raters[r][1] - anchor_raters[r][1] for r in common
            ]
            aroma = mean_ci(aroma_diffs, self.contract["confidence"])
            crisp = mean_ci(crisp_diffs, self.contract["confidence"])
        return Edge(
            session=session,
            cargo=cargo,
            anchor=anchor,
            storage_days=storage_days,
            same_variety=same_variety,
            same_maturity=same_maturity,
            valid_panel=len(common),
            eligible=exclusion is None,
            exclusion=exclusion,
            aroma=aroma,
            crisp=crisp,
            rater_pairs=tuple(common),
        )

    def eligible_edges(self, as_of):
        return tuple(e for e in self.build_edges(as_of) if e.eligible)

    # ------------------------------------------------------------------
    def network_components(self, as_of):
        """以合格边构造无向图（样品为节点），返回连通组件集合列表。

        只有同一组件内的样品才存在可追溯的参照链；断开的组件之间不给绝对差。
        没有任何合格边的样品保留为孤立组件，避免在报告里“静默消失”。
        """
        adjacency: dict[str, set[str]] = {sample: set() for sample in self.samples}
        for edge in self.eligible_edges(as_of):
            adjacency.setdefault(edge.cargo, set()).add(edge.anchor)
            adjacency.setdefault(edge.anchor, set()).add(edge.cargo)
        seen, components = set(), []
        for node in sorted(adjacency):
            if node in seen:
                continue
            stack = [node]
            component = set()
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                component.add(current)
                stack.extend(adjacency[current] - seen)
            components.append(component)
        return components

    def reference_sessions_for(self, cargo, anchor, as_of):
        """货批—参照在不同有效场次可连到的比较；同场重复码只算一场。"""
        return sorted(
            {
                e.session
                for e in self.eligible_edges(as_of)
                if e.cargo == cargo and e.anchor == anchor
            }
        )

    # ------------------------------------------------------------------
    def anchor_presence(self, as_of):
        """每个参照样品出现在哪些仍有效的场次（用于漂移与共同影响分析）。"""
        table: dict[str, list[str]] = {}
        for session in self.sessions:
            for sample in self.session_present_samples(session):
                if self.samples[sample].type != "anchor":
                    continue
                if self.is_invalid(session, sample, as_of):
                    continue
                table.setdefault(sample, []).append(session)
        return {k: tuple(sorted(v)) for k, v in sorted(table.items())}

    def repeatability_records(self):
        """产生同场重复码的 (场次, 样品, 盲码列表, 每码评委评分dict) 记录。"""
        records = []
        for session in self.sessions:
            for sample in self.session_present_samples(session):
                codes = self.duplicate_codes(session, sample)
                if len(codes) < 2:
                    continue
                code_scores = [
                    self.raw_rater_code_scores(session, code) for code in codes
                ]
                records.append((session, sample, codes, code_scores))
        return records

    def rater_cross_session_scores(self):
        """评委 -> 样品 -> [(场次, aroma, crisp)]，只含未失效出场。"""
        table: dict[str, dict[str, list[tuple]]] = {}
        for session in self.sessions:
            for sample in self.session_present_samples(session):
                if self.is_invalid(session, sample, self.today):
                    continue
                for rater, values in self.rater_sample_scores(session, sample).items():
                    table.setdefault(rater, {}).setdefault(sample, []).append(
                        (session, values[0], values[1])
                    )
        return table
