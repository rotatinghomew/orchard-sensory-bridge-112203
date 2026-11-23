"""研究装配、解盲、失效时点、比较边与网络连通性。"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_packet import read_packet
from sensory.study import Study

PACKET = Path(__file__).resolve().parents[1] / "资料" / "演练.json"
TODAY = "2026-11-23"
PRINT_DAY = "2026-10-22"
KNOWN = "2026-11-02"


class StudyAssemblyTest(unittest.TestCase):
    def setUp(self):
        self.study = Study(read_packet(PACKET), TODAY)

    def test_scores_are_unblinded_to_physical_samples(self):
        samples = {score.sample for score in self.study.joined}
        self.assertIn("K1", samples)
        # R1-X3 与 R1-X7 是 AX 的同场重复码。
        codes = self.study.duplicate_codes("R1", "AX")
        self.assertEqual(sorted(codes), ["R1-X3", "R1-X7"])
        rater_scores = self.study.rater_sample_scores("R1", "AX")
        # 同一评委的两个码被合并，重复码数保留。
        self.assertTrue(all(values[2] == 2 for values in rater_scores.values()))
        self.assertEqual(len(rater_scores), 8)

    def test_invalidation_is_time_aware_and_session_scoped(self):
        # 失效 known 之前，R2 的 K1 仍然有效；R1 的 K1 任何时候都有效。
        self.assertIsNone(self.study.is_invalid("R2", "K1", PRINT_DAY))
        self.assertIsNotNone(self.study.is_invalid("R2", "K1", KNOWN))
        self.assertIsNone(self.study.is_invalid("R1", "K1", TODAY))
        edge_before = [
            e for e in self.study.build_edges(PRINT_DAY)
            if e.session == "R2" and e.anchor == "K1"
        ]
        self.assertTrue(all(e.eligible for e in edge_before))
        edge_after = [
            e for e in self.study.build_edges(TODAY)
            if e.session == "R2" and e.anchor == "K1"
        ]
        self.assertTrue(all(not e.eligible for e in edge_after))
        self.assertIn("温控超窗", edge_after[0].exclusion)

    def test_variety_maturity_mismatch_excludes_r4_anchor(self):
        edges = [
            e for e in self.study.build_edges(TODAY)
            if e.session == "R4"
        ]
        self.assertTrue(edges)
        self.assertTrue(all(not e.eligible for e in edges))
        self.assertIn("品种不等价", edges[0].exclusion)

    def test_edges_are_paired_within_session_and_panel_checked(self):
        eligible = self.study.eligible_edges(TODAY)
        self.assertTrue(all(e.valid_panel >= 6 for e in eligible))
        self.assertTrue(all(e.aroma is not None and e.crisp is not None for e in eligible))
        # 同场重复码不得计成两个参照场次。
        ax_k1 = self.study.reference_sessions_for("AX", "K1", TODAY)
        self.assertEqual(ax_k1, ["R1"])

    def test_network_keeps_disconnected_components_separate(self):
        components = self.study.network_components(TODAY)
        joined = {sample for component in components for sample in component}
        self.assertEqual(joined, set(self.study.samples))
        # AV 与 KX 是孤立节点（R4 无合格边），不得并入 Ruby 主组件。
        av_component = next(c for c in components if "AV" in c)
        self.assertEqual(av_component, {"AV"})
        kx_component = next(c for c in components if "KX" in c)
        self.assertEqual(kx_component, {"KX"})
        main = next(c for c in components if {"AZ", "K2", "K3"} <= c)
        self.assertNotIn("TV", main)

    def test_components_before_invalidation_include_r2_k1_path(self):
        before = self.study.network_components(PRINT_DAY)
        main = next(c for c in before if "AY" in c)
        self.assertIn("K1", main)


if __name__ == "__main__":
    unittest.main()
