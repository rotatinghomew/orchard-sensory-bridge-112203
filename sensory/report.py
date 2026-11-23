"""把分析结果装配成运营可阅读的中文报告与机器可读 JSON。"""

from __future__ import annotations

from .claims import (
    INSIDE,
    VERDICT_TEXT,
    assess_label_history,
    assess_request,
)
from .diagnostics import (
    anchor_drift_table,
    category_mismatch_table,
    order_effect_table,
    repeatability_table,
    rater_repeat_table,
)
from .planning import plan_candidates
from .study import Study


def build_view(study: Study, as_of):
    drift_now = anchor_drift_table(study, as_of)
    requests = [
        assess_request(study, request, as_of, drift_now)
        for request in study.label_requests
    ]
    histories = [
        assess_label_history(
            study,
            label,
            as_of,
            anchor_drift_table(study, label["as_of"]),
            drift_now,
        )
        for label in study.labels
    ]
    plans, combinations, conflicts, coverage = plan_candidates(study, as_of)
    return {
        "as_of": str(as_of),
        "edges": study.build_edges(as_of),
        "components": study.network_components(as_of),
        "requests": requests,
        "histories": histories,
        "drift": drift_now,
        "mismatches": category_mismatch_table(study, as_of),
        "repeatability": repeatability_table(study),
        "rater_repeat": rater_repeat_table(study),
        "order_effect": order_effect_table(study),
        "plans": plans,
        "combinations": combinations,
        "conflicts": conflicts,
        "coverage": coverage,
        "invalidation_events": study.invalidation_events(as_of),
    }


def render_markdown(study: Study, as_of):
    view = build_view(study, as_of)
    lines = []
    lines.append("# 冷藏苹果盲评感官比较报告")
    lines.append("")
    lines.append(f"- 判定基准日：{as_of}（失效记录仅在 `known` 当日及以后生效）")
    lines.append(
        f"- 容许带：香气 ±{study.contract['aroma_margin']}，"
        f"脆度 ±{study.contract['crisp_margin']}；"
        f"置信水平 {study.contract['confidence']}"
    )
    lines.append(
        f"- 有效评委下限 {study.contract['min_valid_panel']} 人/场；"
        f"参照场次下限 {study.contract['min_reference_sessions']} 场（同场重复码不算不同场）"
    )
    lines.append("")

    lines.append("## 1. 失效记录与适用时点")
    if view["invalidation_events"]:
        for inv in view["invalidation_events"]:
            lines.append(
                f"- {inv.session} 的 {inv.sample} 自 {inv.known} 起失效：{inv.reason}。"
                "只撤出该场次，其他场次的同一实体参照仍有效。"
            )
    else:
        lines.append("- 基准日前无失效记录。")
    lines.append("")

    lines.append("## 2. 比较边（同场、同品种同熟度、按评委配对的货批−参照差值）")
    lines.append("")
    lines.append("| 场次 | 货批 | 参照 | 日龄 | 评委 | 香气差值区间 | 脆度差值区间 | 状态 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for edge in view["edges"]:
        if edge.eligible:
            status = "有效"
            aroma = edge.aroma.as_text()
            crisp = edge.crisp.as_text()
        else:
            status = f"排除：{edge.exclusion}"
            aroma = crisp = "—"
        lines.append(
            f"| {edge.session} | {edge.cargo} | {edge.anchor} | {edge.storage_days} | "
            f"{edge.valid_panel} | {aroma} | {crisp} | {status} |"
        )
    lines.append("")
    lines.append("可比网络组件（只在同一组件内存在可追溯参照链）：")
    for index, component in enumerate(view["components"], 1):
        if len(component) == 1:
            lines.append(f"- 组件 {index}：{', '.join(sorted(component))}（孤立节点，无合格比较边，网络断开）")
        else:
            lines.append(f"- 组件 {index}（连通组件）：{', '.join(sorted(component))}")
    lines.append("")

    lines.append("## 3. 量表漂移与品类交互诊断")
    lines.append("")
    lines.append("### 3.0 品类/熟度错配（这些配对的刻度架在不等价样品上，不得作锚）")
    if view["mismatches"]:
        lines.append("| 场次 | 货批 | 参照 | 货批品种/熟度 | 参照品种/熟度 | 原因 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in view["mismatches"]:
            lines.append(
                f"| {row.session} | {row.cargo} | {row.anchor} | "
                f"{row.cargo_variety}/{row.cargo_maturity} | "
                f"{row.anchor_variety}/{row.anchor_maturity} | {row.reason} |"
            )
    else:
        lines.append("- 无错配配对。")
    lines.append("")
    lines.append("### 3.1 共享参照跨场漂移（ANOVA，p<0.10 标记）")
    lines.append("| 参照 | 维度 | 场次均值 | 极差 | F | p | 标记 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in view["drift"]:
        lines.append(
            f"| {row.anchor} | {row.attribute} | "
            f"{dict(zip(row.sessions, row.means))} | {row.spread} | "
            f"{row.f_stat} | {row.p_value} | {'漂移' if row.flagged else '稳定'} |"
        )
    lines.append("")
    lines.append("### 3.2 同一评委跨场重复评分（同一实体样品的个人量表波动）")
    lines.append("| 评委 | 样品 | 出场 | 香气SD | 脆度SD |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in view["rater_repeat"]:
        lines.append(
            f"| {row.rater} | {row.sample} | {','.join(row.appearances)} | "
            f"{row.aroma_sd:.2f} | {row.crisp_sd:.2f} |"
        )
    lines.append("")
    lines.append("### 3.3 同批重复样可靠性")
    lines.append("| 场次 | 样品 | 维度 | 重复相关 r | 重复SD | 平均差值 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in view["repeatability"]:
        r_text = "NA" if row.pearson_r is None else f"{row.pearson_r:+.2f}"
        lines.append(
            f"| {row.session} | {row.sample} | {row.attribute} | {r_text} | "
            f"{row.repeat_sd:.2f} | {row.mean_signed_diff:+.2f} |"
        )
    lines.append("")
    lines.append("### 3.4 上样顺序效应")
    for row in view["order_effect"]:
        flag = "存在系统性顺序偏向" if row.flagged else "未见系统偏向（个体相关仍有噪声）"
        lines.append(
            f"- {row.attribute}：{row.rater_sessions} 个评委×场次，"
            f"平均有符号相关 {row.mean_signed_corr:+.3f}，平均绝对相关 {row.mean_abs_corr:.3f} —— {flag}"
        )
    lines.append("")

    lines.append("## 4. 标签请求声明（基准日可写什么）")
    for claim in view["requests"]:
        lines.append("")
        lines.append(f"### {claim.request_id}：货批 {list(claim.items)} × 参照 {list(claim.anchors)}，窗口 {claim.from_day}–{claim.to_day} 日")
        lines.append(f"- 总结论：**{'给通用声明' if claim.general_claim else '不给通用声明'}**。{claim.general_text}")
        for item in claim.item_assessments:
            lines.append(f"- 货批 {item.item}（状态：{item.status}，实测日龄 {list(item.observed_days)}，窗口覆盖日 {list(item.covered_days)}）：{item.local_text}")
            for pair in item.pairs:
                lines.append(
                    f"    - 对 {pair.anchor} @ {pair.storage_days} 日：有效参照场次 "
                    f"{list(pair.reference_sessions) or '无'}（达标：{'是' if pair.session_count_ok else '否'}，"
                    f"连通：{'是' if pair.connected else '否'}）"
                )
                for verdict in (pair.aroma, pair.crisp):
                    if verdict.estimate is None:
                        interval = "无区间"
                    else:
                        interval = verdict.estimate.as_text()
                    lines.append(
                        f"        - {verdict.attribute}（±{verdict.margin}）：{VERDICT_TEXT[verdict.verdict]}；{interval}。{verdict.note}"
                    )
                for note in pair.exclusion_notes:
                    lines.append(f"        - 排除记录：{note}")
                for warning in pair.drift_warnings:
                    lines.append(f"        - 共同影响：{warning}")
        for note in claim.invalidation_notes:
            lines.append(f"- 失效说明：{note}")
        for note in claim.shared_anchor_notes:
            lines.append(f"- 共享参照传播：{note}")
        if claim.unsupported_pairings:
            lines.append("- 未支撑搭配（证据不足，不得写成不合格）：")
            for text in claim.unsupported_pairings:
                lines.append(f"    - {text}")
    lines.append("")

    lines.append("## 5. 已印签的签发依据与范围变化")
    for history in view["histories"]:
        lines.append("")
        lines.append(
            f"### 印签 {history.label_id}（{history.status}，签发日 {history.as_of}）"
        )
        lines.append(f"- 当前是否可继续沿用：**{'是' if history.still_valid else '否，应停止沿用通用表述'}**")
        lines.append(f"- 适用范围是否收缩：{'是' if history.scope_narrowed else '否'}")
        lines.append(f"- 判定说明：{history.reason}")
        for key in history.lost_pairings:
            lines.append(f"    - 失去等同支撑的搭配：{'-'.join(map(str, key))}")
        for key in history.shrunk_pairings:
            lines.append(f"    - 参照场次被削掉的搭配：{'-'.join(map(str, key))}")
        lines.append(
            f"- 签发时通用声明：{'具备' if history.at_print.general_claim else '不具备'}；"
            f"当前通用声明：{'具备' if history.today.general_claim else '不具备'}"
        )
    lines.append("")

    lines.append("## 6. 补评候选场次与留样约束")
    for plan in view["plans"]:
        lines.append("")
        lines.append(f"### 候选 {plan.session_id}（{plan.day}，每个参照至少 {plan.servings_per_anchor} 份）")
        for need in plan.anchor_needs:
            lines.append(
                f"- 参照 {need.anchor}：需 {need.servings_required} 份，"
                f"留样 {need.reserve_available} 份（可用至 {need.keep_until}）—— "
                f"{'可行' if need.feasible else '不可行'}：{need.reason}"
            )
        lines.append(f"- 单场可行性：{'可行' if plan.feasible_alone else '不可行'}。{plan.note}")
        if plan.restorable_pairings:
            lines.append("- 可同时连接并恢复的搭配：")
            for pairing in plan.restorable_pairings:
                lines.append(f"    - {pairing}")
        for link in plan.disallowed_links:
            lines.append(f"- 可连接但未被请求允许：{link}")
        if plan.unconnected_candidates:
            lines.append(f"- 在场内找不到同品种同熟度锚的货批：{', '.join(plan.unconnected_candidates)}")
    lines.append("")
    lines.append("候选同排检查：")
    for left, right, shared in view["combinations"]:
        shared_text = "无共享参照" if not shared else f"共享参照 {', '.join(shared)} 留样充足"
        lines.append(f"- {left} + {right}：可同排（{shared_text}）")
    for left, right, reasons in view["conflicts"]:
        lines.append(f"- {left} + {right}：不可同排")
        for reason in reasons:
            lines.append(f"    - {reason}")
    lines.append("")
    lines.append("按请求汇总的恢复路径：")
    for request_id, paths in sorted(view["coverage"].items()):
        for path in sorted(paths):
            lines.append(f"- {request_id}：{path}")
    lines.append("")
    lines.append("> 货批留样由候选场次组织者自备，不占参照果留样；任何补评都不能用同一份已耗用留样补两个场次。")
    return "\n".join(lines) + "\n"
