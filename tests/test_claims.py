"""声明判定：容许带三分类、参照场次、窗口覆盖、通用/局部、印签历史。"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_packet import read_packet
from sensory.claims import (
    CROSSES,
    INSIDE,
    OUTSIDE,
    UNKNOWN,
    assess_label_history,
    assess_request,
)
from sensory.diagnostics import anchor_drift_table
from sensory.study import Study

PACKET = Path(__file__).resolve().parents[1] / "资料" / "演练.json"
TODAY = "2026-11-23"
PRINT_DAY = "2026-10-22"


class ClaimTest(unittest.TestCase):
    def setUp(self):
        self.study = Study(read_packet(PACKET), TODAY)
        self.drift = anchor_drift_table(self.study, TODAY)
        self.requests = {r["id"]: r for r in self.study.label_requests}

    def assess(self, request_id, as_of=TODAY):
        return assess_request(
            self.study, self.requests[request_id], as_of,
            anchor_drift_table(self.study, as_of),
        )

    def pair(self, assessment, item, anchor, day):
        item_assessment = next(ia for ia in assessment.item_assessments if ia.item == item)
        return next(
            pa for pa in item_assessment.pairs
            if pa.anchor == anchor and pa.storage_days == day
        )

    def test_l2_general_claim_supported_with_two_reference_sessions(self):
        claim = self.assess("L2")
        self.assertTrue(claim.general_claim)
        for item in ("AZ", "AW"):
            for anchor in ("K2", "K3"):
                pair = self.pair(claim, item, anchor, 70)
                self.assertEqual(pair.reference_sessions, ("R3", "R6"))
                self.assertTrue(pair.session_count_ok)
                self.assertEqual(pair.aroma.verdict, INSIDE)
                self.assertEqual(pair.crisp.verdict, INSIDE)
                self.assertGreaterEqual(pair.aroma.estimate.low, -0.65)
                self.assertLessEqual(pair.aroma.estimate.high, 0.65)

    def test_l1_fails_on_reference_count_not_on_outside_verdict(self):
        claim = self.assess("L1")
        self.assertFalse(claim.general_claim)
        for item in ("AX", "AY"):
            pair = self.pair(claim, item, "K1", 55)
            self.assertFalse(pair.session_count_ok)
            self.assertEqual(pair.reference_sessions, ("R1",))
            self.assertEqual(pair.aroma.verdict, UNKNOWN)
            self.assertEqual(pair.crisp.verdict, UNKNOWN)
        # 没有任何搭配被判成 OUTSIDE：缺证不等于不合格。
        verdicts = [
            v.verdict
            for ia in claim.item_assessments for pa in ia.pairs
            for v in (pa.aroma, pa.crisp)
        ]
        self.assertNotIn(OUTSIDE, verdicts)

    def test_l1_at_print_had_two_sessions_for_ay_then_lost_one(self):
        at_print = self.assess("L1", PRINT_DAY)
        pair_print = self.pair(at_print, "AY", "K1", 55)
        self.assertEqual(pair_print.reference_sessions, ("R1", "R2"))
        # 两场仍未把区间收进容许带：签发时同样不能给通用声明。
        self.assertFalse(at_print.general_claim)
        self.assertIn(pair_print.crisp.verdict, (CROSSES, UNKNOWN))
        pair_now = self.pair(self.assess("L1"), "AY", "K1", 55)
        self.assertEqual(pair_now.reference_sessions, ("R1",))

    def test_l3_disconnected_and_single_session_are_not_failures(self):
        claim = self.assess("L3")
        self.assertFalse(claim.general_claim)
        av = next(ia for ia in claim.item_assessments if ia.item == "AV")
        self.assertEqual(av.status, "disconnected")
        tv = next(ia for ia in claim.item_assessments if ia.item == "TV")
        self.assertEqual(tv.status, "insufficient")
        # 90 日单点只能为自己作证，不能外推到 80–95 整个窗口。
        self.assertEqual(tv.covered_days, (90,))
        pair = self.pair(claim, "TV", "KY", 90)
        self.assertEqual(pair.reference_sessions, ("R5",))
        self.assertFalse(pair.session_count_ok)

    def test_window_interpolation_but_no_extrapolation(self):
        # 构造一个有两端实测的请求，中间日应被覆盖；窗口外不外推。
        request = {
            "id": "LX", "items": ["AZ"], "anchors": ["K2", "K3"],
            "from_day": 70, "to_day": 72,
        }
        claim = assess_request(self.study, request, TODAY, self.drift)
        item = claim.item_assessments[0]
        self.assertEqual(item.observed_days, (70,))
        self.assertEqual(item.covered_days, (70,))
        self.assertFalse(item.window_covered)

    def test_local_status_requires_actual_supported_pair(self):
        # 70–72 日窗口只有 70 日有证据，而 70 日的 AZ 搭配全部达标：
        # 应给 local（局部等同文字 + 明确范围），而不是笼统的 insufficient。
        request = {
            "id": "LX", "items": ["AZ"], "anchors": ["K2", "K3"],
            "from_day": 70, "to_day": 72,
        }
        claim = assess_request(self.study, request, TODAY, self.drift)
        item = claim.item_assessments[0]
        self.assertEqual(item.status, "local")
        self.assertIn("仅", item.local_text)
        self.assertIn("不能外推", item.local_text)
        # 对照：L1 的两个货批没有任何达标搭配，必须是 insufficient。
        l1 = self.assess("L1")
        self.assertTrue(all(ia.status == "insufficient" for ia in l1.item_assessments))

    def test_shared_anchor_drift_propagates_to_every_dependent_claim(self):
        k3_warning_anchors = set()
        for request_id in self.requests:
            claim = self.assess(request_id)
            for note in claim.shared_anchor_notes:
                self.assertIn("K3", note)
                k3_warning_anchors.add(request_id)
        self.assertIn("L2", k3_warning_anchors)

    def test_label_history_l1_stopped_l2_kept(self):
        labels = {row["id"]: row for row in self.study.labels}
        l1 = assess_label_history(
            self.study, labels["L1"], TODAY,
            anchor_drift_table(self.study, PRINT_DAY), self.drift,
        )
        self.assertFalse(l1.still_valid)
        self.assertTrue(l1.scope_narrowed)
        self.assertIn("参照场次", l1.reason)
        l2 = assess_label_history(
            self.study, labels["L2"], TODAY,
            anchor_drift_table(self.study, PRINT_DAY), self.drift,
        )
        self.assertTrue(l2.still_valid)
        self.assertFalse(l2.scope_narrowed)
        self.assertTrue(l2.at_print.general_claim and l2.today.general_claim)


if __name__ == "__main__":
    unittest.main()
