"""乐句出现检测（phrase occurrence）纯算法与接口测试。"""

import os
import tempfile
import unittest

# 独立运行时为 app.database 提供临时数据库；与 test_api 一起被 discover
# 时复用其已设置的路径（两个测试文件都导入 app.main，路径只可设置一次）
_TMPDIR = tempfile.mkdtemp(prefix="beat-align-test-phrases-")
os.environ.setdefault("ALIGN_DB_PATH", os.path.join(_TMPDIR, "test.db"))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.phrases import PhraseError, find_occurrences  # noqa: E402

REF4 = [0.0, 0.5, 1.0, 1.5]  # 4 拍短乐句
REF3 = [0.0, 0.5, 1.0]  # 3 拍短乐句


class TestFindOccurrences(unittest.TestCase):
    def test_exact_single_occurrence(self):
        out = find_occurrences(REF4, [0.0, 0.5, 1.0, 1.5], [1.0], 0.05)
        self.assertEqual(out["occurrence_count"], 1)
        self.assertEqual(out["candidate_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["rate"], 1.0)
        self.assertEqual(occ["rate_index"], 0)
        self.assertEqual(occ["start_time"], 0.0)
        self.assertEqual(occ["end_time"], 1.5)
        self.assertEqual(occ["start_onset_index"], 0)
        self.assertEqual(occ["end_onset_index"], 3)
        self.assertEqual(occ["matched_count"], 4)
        self.assertEqual(occ["total_absolute_error"], 0.0)
        self.assertEqual(
            [(m["ref_index"], m["onset_index"]) for m in occ["mapping"]],
            [(0, 0), (1, 1), (2, 2), (3, 3)],
        )
        self.assertTrue(all(m["error"] == 0.0 for m in occ["mapping"]))

    def test_rate_scaling_selects_matching_rate(self):
        # 录音以半速演奏乐句：拍间隔 1.0s = 0.5s × 倍率 2.0
        onsets = [10.0, 11.0, 12.0, 13.0]
        out = find_occurrences(REF4, onsets, [1.0, 2.0], 0.05)
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["rate"], 2.0)
        self.assertEqual(occ["rate_index"], 1)
        self.assertEqual(occ["start_time"], 10.0)
        self.assertEqual(occ["end_time"], 13.0)
        self.assertEqual(occ["matched_count"], 4)

        # 只给倍率 1.0 -> 无命中
        out = find_occurrences(REF4, onsets, [1.0], 0.05)
        self.assertEqual(out["occurrences"], [])

    def test_missed_beat_and_extra_onsets(self):
        # 录音多出一个起音 0.3（多余起音），且参考第 3 拍（1.0s）被漏
        onsets = [0.0, 0.3, 0.5, 1.5]
        out = find_occurrences(REF4, onsets, [1.0], 0.05)
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["matched_count"], 3)
        self.assertEqual(
            [(m["ref_index"], m["onset_index"]) for m in occ["mapping"]],
            [(0, 0), (1, 2), (3, 3)],
        )

    def test_fewer_than_three_matches_rejected(self):
        # 仅首末两拍可匹配（总匹配 2 < 3）-> 无命中
        out = find_occurrences(REF4, [0.0, 1.5], [1.0], 0.05)
        self.assertEqual(out["occurrences"], [])
        # 只有 2 拍的乐句永远凑不足 3 拍 -> 无命中
        out = find_occurrences([0.0, 1.0], [0.0, 1.0], [1.0], 0.05)
        self.assertEqual(out["occurrences"], [])

    def test_tolerance_boundary(self):
        onsets = [0.0, 0.6, 1.0]  # 中间拍误差恰为 0.1s
        out = find_occurrences(REF3, onsets, [1.0], 0.1)
        self.assertEqual(out["occurrence_count"], 1)
        self.assertAlmostEqual(
            out["occurrences"][0]["total_absolute_error"], 0.1, places=6
        )
        # 容差收紧到 0.09 -> 中间拍不可匹配，总匹配不足 3 拍
        out = find_occurrences(REF3, onsets, [1.0], 0.09)
        self.assertEqual(out["occurrences"], [])

    def test_same_span_prefers_more_matches_then_lower_error(self):
        # 0.48 / 0.53 都在参考第 2 拍（0.5s）容差窗内：
        # 同一（首, 尾, 倍率）下应选 4 拍全匹配、且总误差更小的 0.48
        onsets = [0.0, 0.48, 0.53, 1.0, 1.5]
        out = find_occurrences(REF4, onsets, [1.0], 0.05)
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["matched_count"], 4)
        self.assertEqual(
            [(m["ref_index"], m["onset_index"]) for m in occ["mapping"]],
            [(0, 0), (1, 1), (2, 3), (3, 4)],
        )
        self.assertAlmostEqual(occ["total_absolute_error"], 0.02, places=6)

    def test_global_selection_maximizes_non_overlapping_count(self):
        # 候选 [0..2] 与 [1..3] 互相重叠，[4..6] 与两者都不重叠：
        # 最多保留 2 次出现；次数与总误差并列时取起点序列更早的 [0..2]
        onsets = [0.0, 0.5, 1.0, 1.5, 3.0, 3.5, 4.0]
        out = find_occurrences(REF3, onsets, [1.0], 0.05)
        self.assertEqual(out["occurrence_count"], 2)
        self.assertEqual(
            [(o["start_onset_index"], o["end_onset_index"]) for o in out["occurrences"]],
            [(0, 2), (4, 6)],
        )

    def test_global_selection_minimizes_total_error(self):
        # 候选 A=[0..2]（误差 0.09）与 B=[1..4]（误差 0.02）重叠，只能留一个：
        # 出现次数并列时取总绝对误差更小的 B（尽管起点更晚）
        onsets = [0.0, 0.59, 1.0, 1.08, 1.60]
        out = find_occurrences(REF3, onsets, [1.0], 0.1)
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["start_onset_index"], 1)
        self.assertEqual(occ["end_onset_index"], 4)
        self.assertEqual(occ["matched_count"], 3)
        self.assertAlmostEqual(occ["total_absolute_error"], 0.02, places=6)
        self.assertEqual(
            [(m["ref_index"], m["onset_index"]) for m in occ["mapping"]],
            [(0, 1), (1, 3), (2, 4)],
        )

    def test_touching_intervals_count_as_overlap(self):
        # 三次出现首尾相接（前一次的尾起音 = 后一次的首起音）：
        # 端点相接视为重叠，最多保留 1 次，并列取起点最早者
        onsets = [0.0, 0.5, 1.0, 1.5, 2.0]
        out = find_occurrences(REF3, onsets, [1.0], 0.05)
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["start_onset_index"], 0)
        self.assertEqual(occ["end_onset_index"], 2)

    def test_rate_input_order_breaks_ties(self):
        # 两个相同倍率产生完全并列的候选 -> 取输入顺序靠前者（rate_index 0）
        out = find_occurrences(REF3, [0.0, 0.5, 1.0], [1.0, 1.0], 0.05)
        self.assertEqual(out["occurrence_count"], 1)
        self.assertEqual(out["occurrences"][0]["rate_index"], 0)
        # 只有靠后的倍率能命中时，返回其输入序号
        out = find_occurrences(REF3, [0.0, 0.5, 1.0], [2.0, 1.0], 0.05)
        self.assertEqual(out["occurrence_count"], 1)
        self.assertEqual(out["occurrences"][0]["rate_index"], 1)

    def test_multiple_rates_find_multiple_occurrences(self):
        # 录音先后以原速与半速各演奏一次乐句
        onsets = [0.0, 0.5, 1.0, 3.0, 4.0, 5.0]
        out = find_occurrences(REF3, onsets, [1.0, 2.0], 0.05)
        self.assertEqual(out["occurrence_count"], 2)
        self.assertEqual([o["rate"] for o in out["occurrences"]], [1.0, 2.0])
        self.assertEqual(
            [(o["start_time"], o["end_time"]) for o in out["occurrences"]],
            [(0.0, 1.0), (3.0, 5.0)],
        )

    def test_no_hit_returns_empty(self):
        out = find_occurrences(REF4, [0.0, 0.13, 0.77, 2.9], [1.0], 0.05)
        self.assertEqual(out["occurrence_count"], 0)
        self.assertEqual(out["occurrences"], [])

    def test_invalid_inputs_raise(self):
        with self.assertRaises(PhraseError):
            find_occurrences([0.0, 1.0, 0.5], [0.0, 1.0], [1.0], 0.1)  # 非递增
        with self.assertRaises(PhraseError):
            find_occurrences([0.0, 1.0], [0.0, 1.0], [0.0], 0.1)  # 倍率非正
        with self.assertRaises(PhraseError):
            find_occurrences([0.0, 1.0], [0.0, 1.0], [1.0], -0.1)  # 容差为负
        with self.assertRaises(PhraseError):
            find_occurrences([0.0, 1.0], [0.0, 1.0], [], 0.1)  # 无倍率
        with self.assertRaises(PhraseError):
            find_occurrences([0.0], [0.0, 1.0], [1.0], 0.1)  # 参考拍不足 2 个


class TestPhraseOccurrencesAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()
        cls.addClassCleanup(cls._cm.__exit__, None, None, None)

    def test_endpoint_basic_flow(self):
        r = self.client.post("/phrases/occurrences", json={
            "reference_beats": REF4,
            "recording_onsets": [0.0, 0.5, 1.0, 1.5, 3.0, 4.0, 5.0, 6.0],
            "rates": [1.0, 2.0],
            "tolerance": 0.05,
        })
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()
        self.assertEqual(out["occurrence_count"], 2)
        self.assertEqual(out["parameters"]["tolerance"], 0.05)
        self.assertEqual(out["parameters"]["rates"], [1.0, 2.0])
        first, second = out["occurrences"]
        self.assertEqual((first["rate"], first["start_time"], first["end_time"]),
                         (1.0, 0.0, 1.5))
        self.assertEqual((second["rate"], second["start_time"], second["end_time"]),
                         (2.0, 3.0, 6.0))
        self.assertEqual(
            [(m["ref_index"], m["onset_index"]) for m in second["mapping"]],
            [(0, 4), (1, 5), (2, 6), (3, 7)],
        )

    def test_endpoint_empty_occurrences(self):
        r = self.client.post("/phrases/occurrences", json={
            "reference_beats": REF4,
            "recording_onsets": [0.0, 0.13, 0.77, 2.9],
            "rates": [1.0],
            "tolerance": 0.05,
        })
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["occurrences"], [])

    def test_endpoint_validation_422(self):
        base = {
            "reference_beats": REF4,
            "recording_onsets": [0.0, 0.5, 1.0, 1.5],
            "rates": [1.0],
            "tolerance": 0.05,
        }
        bad_payloads = [
            {**base, "reference_beats": [0.0, 1.0, 0.5, 2.0]},  # 参考拍非递增
            {**base, "recording_onsets": [0.0, 0.5, 0.5, 1.0]},  # 起音非递增
            {**base, "rates": [1.0, 0.0]},  # 倍率非正
            {**base, "rates": [-1.0]},
            {**base, "rates": []},  # 空倍率列表
            {**base, "tolerance": -0.5},  # 容差为负
            {**base, "reference_beats": [0.0]},  # 参考拍不足 2 个
            {**base, "recording_onsets": []},  # 空起音列表
        ]
        for payload in bad_payloads:
            r = self.client.post("/phrases/occurrences", json=payload)
            self.assertEqual(r.status_code, 422, payload)


if __name__ == "__main__":
    unittest.main()
