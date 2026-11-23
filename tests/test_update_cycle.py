"""资料包可反复更新：追加场次与扣减留样后，结论自动重算。"""

import copy
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_packet import read_packet
from sensory.claims import assess_request
from sensory.diagnostics import anchor_drift_table
from sensory.planning import plan_candidates
from sensory.study import Study

PACKET = Path(__file__).resolve().parents[1] / "资料" / "演练.json"
TODAY = "2026-11-23"


class UpdateCycleTest(unittest.TestCase):
    def setUp(self):
        self.packet = read_packet(PACKET)

    def _append_session(self, packet, session_id, cargo, anchor, day, difference=0.0):
        packet["procedures"].append(
            {"id": session_id, "day": TODAY, "panel": session_id}
        )
        packet["blind_map"] += [
            {"session": session_id, "code": f"{session_id}-C", "sample": cargo,
             "reveal": TODAY, "storage_days": day},
            {"session": session_id, "code": f"{session_id}-A", "sample": anchor,
             "reveal": TODAY, "storage_days": 0},
        ]
        for index in range(8):
            rater = f"Q{index}"
            jitter = (index - 3.5) * 0.05
            packet["scores"] += [
                {"session": session_id, "rater": rater, "code": f"{session_id}-C",
                 "aroma": 5.0 + difference + jitter, "crisp": 5.0 + difference - jitter,
                 "serving_order": 1},
                {"session": session_id, "rater": rater, "code": f"{session_id}-A",
                 "aroma": 5.0 - jitter, "crisp": 5.0 + jitter, "serving_order": 2},
            ]

    def test_new_session_restores_reference_count(self):
        study = Study(copy.deepcopy(self.packet), TODAY)
        request = next(r for r in study.label_requests if r["id"] == "L1")
        before = assess_request(
            study, request, TODAY, anchor_drift_table(study, TODAY)
        )
        ax_before = next(ia for ia in before.item_assessments if ia.item == "AX")
        self.assertEqual(ax_before.pairs[0].reference_sessions, ("R1",))

        updated = copy.deepcopy(self.packet)
        self._append_session(updated, "N1", "AX", "K1", 55)
        study2 = Study(updated, TODAY)
        after = assess_request(
            study2, request, TODAY, anchor_drift_table(study2, TODAY)
        )
        ax_after = next(ia for ia in after.item_assessments if ia.item == "AX")
        self.assertEqual(ax_after.pairs[0].reference_sessions, ("N1", "R1"))
        self.assertTrue(ax_after.pairs[0].session_count_ok)

    def test_reserve_consumption_blocks_later_candidate(self):
        updated = copy.deepcopy(self.packet)
        k1 = next(row for row in updated["reserves"] if row["sample"] == "K1")
        k1["servings"] = 4  # 已被上一场耗用 8 份
        study = Study(updated, TODAY)
        plans, _combinations, _conflicts, _coverage = plan_candidates(study, TODAY)
        n2 = next(plan for plan in plans if plan.session_id == "N2")
        self.assertFalse(n2.feasible_alone)
        reason = " ".join(need.reason for need in n2.anchor_needs if need.anchor == "K1")
        self.assertIn("仅剩 4 份", reason)


if __name__ == "__main__":
    unittest.main()
