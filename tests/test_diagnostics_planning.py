"""诊断与补评规划：漂移、重复样、顺序效应、留样冲突与恢复连接。"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_packet import read_packet
from sensory.diagnostics import (
    anchor_drift_table,
    category_mismatch_table,
    order_effect_table,
    rater_repeat_table,
    repeatability_table,
)
from sensory.planning import plan_candidates
from sensory.study import Study

PACKET = Path(__file__).resolve().parents[1] / "资料" / "演练.json"
TODAY = "2026-11-23"


class DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.study = Study(read_packet(PACKET), TODAY)

    def test_k3_crisp_drift_flagged_and_k2_stable(self):
        rows = anchor_drift_table(self.study, TODAY)
        k3_crisp = next(r for r in rows if r.anchor == "K3" and r.attribute == "crisp")
        self.assertTrue(k3_crisp.flagged)
        self.assertLess(k3_crisp.p_value, 0.10)
        k2 = [r for r in rows if r.anchor == "K2"]
        self.assertTrue(all(not r.flagged for r in k2))

    def test_repeatability_covers_all_duplicate_sets(self):
        rows = repeatability_table(self.study)
        pairs = {(r.session, r.sample) for r in rows}
        self.assertEqual(
            pairs,
            {("R1", "AX"), ("R2", "AY"), ("R3", "AZ"), ("R4", "AW"), ("R5", "TV")},
        )
        # 每个重复集都有香气与脆度两行，且相关值在 [-1,1]。
        for row in rows:
            self.assertIsNotNone(row.pearson_r)
            self.assertTrue(-1 <= row.pearson_r <= 1)
            self.assertGreater(row.repeat_sd, 0)

    def test_rater_repeat_links_same_person_across_sessions(self):
        rows = rater_repeat_table(self.study)
        p4 = [r for r in rows if r.rater == "P4"]
        self.assertTrue(any(len(r.appearances) >= 3 for r in p4))
        # 失效场次的 K1 出场不得进入重复评分表。
        k1_rows = [r for r in rows if r.sample == "K1"]
        self.assertTrue(all("R2" not in r.appearances for r in k1_rows))

    def test_order_effect_no_systematic_bias(self):
        rows = order_effect_table(self.study)
        self.assertEqual({r.attribute for r in rows}, {"aroma", "crisp"})
        self.assertTrue(all(not r.flagged for r in rows))

    def test_category_mismatch_lists_r4_honey_anchor(self):
        rows = category_mismatch_table(self.study, TODAY)
        self.assertTrue(rows)
        self.assertTrue(all(row.session == "R4" for row in rows))
        self.assertTrue({row.anchor for row in rows} == {"KX"})
        self.assertTrue(all("品种" in row.reason for row in rows))
        # 错配只出现在诊断表，不产生任何合格比较边。
        self.assertFalse(
            any(e.eligible for e in self.study.build_edges(TODAY) if e.session == "R4")
        )


class PlanningTest(unittest.TestCase):
    def setUp(self):
        self.study = Study(read_packet(PACKET), TODAY)
        self.plans, self.combinations, self.conflicts, self.coverage = plan_candidates(
            self.study, TODAY
        )

    def test_n3_blocked_by_keep_until(self):
        n3 = next(p for p in self.plans if p.session_id == "N3")
        self.assertFalse(n3.feasible_alone)
        self.assertTrue(all(not need.feasible for need in n3.anchor_needs))
        for need in n3.anchor_needs:
            self.assertIn("2026-11-23", need.reason)

    def test_n1_n2_each_feasible_but_conflict_on_shared_k1(self):
        by_id = {p.session_id: p for p in self.plans}
        self.assertTrue(by_id["N1"].feasible_alone)
        self.assertTrue(by_id["N2"].feasible_alone)
        self.assertEqual(self.combinations, [])
        self.assertEqual(len(self.conflicts), 1)
        left, right, reasons = self.conflicts[0]
        self.assertEqual({left, right}, {"N1", "N2"})
        self.assertTrue(any("K1" in reason for reason in reasons))

    def test_plans_connect_requested_pairings_and_flag_disallowed_links(self):
        by_id = {p.session_id: p for p in self.plans}
        restorable = " ".join(by_id["N1"].restorable_pairings)
        self.assertIn("AX-K1", restorable)
        self.assertIn("AZ-K2", restorable)
        disallowed = " ".join(by_id["N3"].disallowed_links)
        self.assertIn("AV-K2", disallowed)
        # L3 的恢复路径只能出现在不可行的 N3 上。
        self.assertTrue(
            all(paths == set() or any("N3" in p for p in paths)
                for rid, paths in self.coverage.items() if rid == "L3")
        )

    def test_keep_until_day_is_still_usable(self):
        # 同一天举行的假设场次应当通过 keep_until 检查（当日可用）。
        from datetime import date
        plans, _, _, _ = plan_candidates(self.study, date(2026, 11, 23))
        n1 = next(p for p in plans if p.session_id == "N1")
        self.assertTrue(n1.feasible_alone)


if __name__ == "__main__":
    unittest.main()
