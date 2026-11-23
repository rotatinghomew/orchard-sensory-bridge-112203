"""命令行入口：运营查询批次/窗口声明、印签依据、补评留样原因。

用法（从项目根目录）：
    python3 sensory_cli.py report [资料路径] [--as-of 2026-11-23] [--out 报告.md]
    python3 sensory_cli.py request L2 [--as-of 2026-11-23]
    python3 sensory_cli.py label L1
    python3 sensory_cli.py sessions
    python3 sensory_cli.py network
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# 以 python -I 运行时脚本目录不会自动进 sys.path，显式加入项目根。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_packet import read_packet
from sensory.claims import VERDICT_TEXT, assess_label_history, assess_request
from sensory.diagnostics import anchor_drift_table
from sensory.planning import plan_candidates
from sensory.report import render_markdown
from sensory.study import Study

DEFAULT_PACKET = Path("资料/演练.json")


def _study(path, as_of):
    packet = read_packet(Path(path))
    return Study(packet, as_of)


def _print_request(study, request_id, as_of):
    known = {r["id"] for r in study.label_requests}
    if request_id not in known:
        raise SystemExit(f"未知标签请求编号：{request_id}；可选 {sorted(known)}")
    request = next(r for r in study.label_requests if r["id"] == request_id)
    drift = anchor_drift_table(study, as_of)
    claim = assess_request(study, request, as_of, drift)
    print(f"标签请求 {claim.request_id}（窗口 {claim.from_day}–{claim.to_day} 日，基准日 {as_of}）")
    print(f"通用声明：{'可以' if claim.general_claim else '不可以'}")
    print(claim.general_text)
    for item in claim.item_assessments:
        print(f"\n货批 {item.item} [{item.status}] 实测日龄={list(item.observed_days)} 覆盖日={list(item.covered_days)}")
        print(f"  {item.local_text}")
        for pair in item.pairs:
            print(
                f"  - 对 {pair.anchor}@{pair.storage_days}日 参照场次={list(pair.reference_sessions)} "
                f"场次达标={pair.session_count_ok} 连通={pair.connected}"
            )
            for verdict in (pair.aroma, pair.crisp):
                interval = "无区间" if verdict.estimate is None else verdict.estimate.as_text()
                print(
                    f"      {verdict.attribute}: {VERDICT_TEXT[verdict.verdict]} | {interval} | {verdict.note}"
                )
            for warning in pair.drift_warnings:
                print(f"      共同影响: {warning}")
    for note in claim.unsupported_pairings:
        print(f"未支撑（不等于不合格）: {note}")


def _print_label(study, label_id, as_of):
    known = {row["id"] for row in study.labels}
    if label_id not in known:
        raise SystemExit(f"未知印签编号：{label_id}；可选 {sorted(known)}")
    label = next(row for row in study.labels if row["id"] == label_id)
    history = assess_label_history(
        study,
        label,
        as_of,
        anchor_drift_table(study, label["as_of"]),
        anchor_drift_table(study, as_of),
    )
    print(f"印签 {history.label_id}（{history.status}，签发日 {history.as_of}）")
    print(f"当前可沿用: {history.still_valid} | 范围收缩: {history.scope_narrowed}")
    print(history.reason)
    print("\n签发时点：")
    print(" ", history.at_print.general_text)
    for item in history.at_print.item_assessments:
        print(f"  {item.item}: {item.local_text}")
    print("\n当前时点：")
    print(" ", history.today.general_text)
    for item in history.today.item_assessments:
        print(f"  {item.item}: {item.local_text}")


def _print_sessions(study, as_of):
    plans, combinations, conflicts, coverage = plan_candidates(study, as_of)
    for plan in plans:
        print(f"候选 {plan.session_id}（{plan.day}）单场可行={plan.feasible_alone}：{plan.note}")
        for need in plan.anchor_needs:
            print(
                f"  锚 {need.anchor}: 需{need.servings_required} 留{need.reserve_available}"
                f"（至{need.keep_until}）可行={need.feasible} — {need.reason}"
            )
        for pairing in plan.restorable_pairings:
            print(f"  可恢复: {pairing}")
        for link in plan.disallowed_links:
            print(f"  可连接但未允许: {link}")
        if plan.unconnected_candidates:
            print(f"  无法连接同品种同熟度锚: {', '.join(plan.unconnected_candidates)}")
    print("\n同排组合：")
    for left, right, shared in combinations:
        print(f"  {left}+{right} 可同排 共享={list(shared)}")
    for left, right, reasons in conflicts:
        print(f"  {left}+{right} 冲突: {'; '.join(reasons)}")
    print("\n为什么下一场要带这些留样：")
    for request_id, paths in sorted(coverage.items()):
        for path in sorted(paths):
            print(f"  {request_id}: {path}")


def _print_network(study, as_of):
    print(f"基准日 {as_of} 的比较边：")
    for edge in study.build_edges(as_of):
        if edge.eligible:
            print(
                f"  {edge.session} {edge.cargo}-{edge.anchor} d{edge.storage_days} "
                f"n={edge.valid_panel} 香气{edge.aroma.as_text()} 脆度{edge.crisp.as_text()}"
            )
        else:
            print(f"  {edge.session} {edge.cargo}-{edge.anchor} 排除: {edge.exclusion}")
    print("\n连通组件：")
    for index, component in enumerate(study.network_components(as_of), 1):
        print(f"  组件{index}: {', '.join(sorted(component))}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="冷藏苹果盲评感官比较")
    parser.add_argument("command", choices=["report", "request", "label", "sessions", "network"])
    parser.add_argument("target", nargs="?", default=None)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--as-of", default=str(date.today()))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    study = _study(args.packet, args.as_of)
    if args.command == "report":
        markdown = render_markdown(study, args.as_of)
        if args.out:
            args.out.write_text(markdown, encoding="utf-8")
            print(f"已写入 {args.out}")
        else:
            print(markdown)
    elif args.command == "request":
        if not args.target:
            parser.error("request 需要标签请求编号，如 L2")
        _print_request(study, args.target, args.as_of)
    elif args.command == "label":
        if not args.target:
            parser.error("label 需要印签编号，如 L1")
        _print_label(study, args.target, args.as_of)
    elif args.command == "sessions":
        _print_sessions(study, args.as_of)
    elif args.command == "network":
        _print_network(study, args.as_of)


if __name__ == "__main__":
    main()
