"""乐句出现搜索（phrase occurrence search）纯算法单元测试。"""

import unittest

from app.phrase import PhraseError, find_phrase_occurrences

PHRASE = [0.0, 0.5, 1.0, 1.5]  # 4 拍短乐句，拍间隔 0.5s，跨度 1.5s


def _search(ref=PHRASE, onsets=(), ratios=(1.0,), tolerance=0.05):
    return find_phrase_occurrences(
        list(ref), list(onsets), list(ratios), tolerance
    )


class TestPhraseOccurrences(unittest.TestCase):
    def test_exact_match_at_ratio_one(self):
        # 录音：前置噪声起音 + 完整乐句（偏移 2.0）+ 尾部噪声
        onsets = [0.3, 2.0, 2.5, 3.0, 3.5, 9.9]
        out = _search(onsets=onsets)
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["ratio"], 1.0)
        self.assertEqual(occ["ratio_index"], 0)
        self.assertEqual(occ["start_time"], 2.0)
        self.assertEqual(occ["end_time"], 3.5)
        self.assertEqual(occ["start_onset_index"], 1)
        self.assertEqual(occ["end_onset_index"], 4)
        self.assertEqual(occ["matched_count"], 4)
        self.assertEqual(occ["total_abs_error"], 0.0)
        self.assertEqual(
            [(m["ref_index"], m["onset_index"]) for m in occ["mapping"]],
            [(0, 1), (1, 2), (2, 3), (3, 4)],
        )
        self.assertTrue(all(m["error"] == 0.0 for m in occ["mapping"]))

    def test_scaled_match_at_ratio_two(self):
        # 倍率 2.0：实际拍间隔 1.0s = 2 × 参考 0.5s
        onsets = [1.0, 2.0, 3.0, 4.0]
        out = _search(onsets=onsets, ratios=(2.0,))
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["ratio"], 2.0)
        self.assertEqual(occ["start_time"], 1.0)
        self.assertEqual(occ["end_time"], 4.0)
        self.assertEqual(occ["matched_count"], 4)

    def test_missed_beat_allowed_but_keeps_three_matches(self):
        # 参考第 2 拍（1.0s）漏拍：剩余 3 拍匹配仍构成出现
        onsets = [0.0, 0.5, 1.5]
        out = _search(onsets=onsets)
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["matched_count"], 3)
        self.assertEqual(
            [(m["ref_index"], m["onset_index"]) for m in occ["mapping"]],
            [(0, 0), (1, 1), (3, 2)],
        )

    def test_fewer_than_three_matches_rejected(self):
        # 仅首尾两拍匹配（中间全漏）-> 不足 3 拍，不出现
        onsets = [0.0, 0.2, 1.5]
        out = _search(onsets=onsets)
        self.assertEqual(out["occurrence_count"], 0)
        self.assertEqual(out["candidate_count"], 0)
        self.assertEqual(out["occurrences"], [])

    def test_extra_onsets_ignored(self):
        # 乐句中间插入多余起音，不影响匹配
        onsets = [0.0, 0.2, 0.5, 0.7, 1.0, 1.3, 1.5]
        out = _search(onsets=onsets)
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["matched_count"], 4)
        self.assertEqual(
            [m["onset_index"] for m in occ["mapping"]], [0, 2, 4, 6]
        )

    def test_first_and_last_reference_beats_must_match(self):
        # 起音只覆盖中间参考拍（首拍无对应起音）-> 不出现
        onsets = [0.5, 1.0, 1.6]
        out = _search(onsets=onsets)
        self.assertEqual(out["occurrence_count"], 0)

    def test_tolerance_boundary_inclusive(self):
        # 误差恰等于容差 -> 可匹配；严格超过 -> 该拍不可匹配
        onsets = [0.0, 0.55, 1.0, 1.5]  # 第 1 拍误差 +0.05
        out = _search(onsets=onsets, tolerance=0.05)
        self.assertEqual(out["occurrence_count"], 1)
        self.assertEqual(out["occurrences"][0]["matched_count"], 4)

        out = _search(onsets=onsets, tolerance=0.049)
        # 第 1 拍不可匹配，但剩余 3 拍仍满足 >= 3
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["matched_count"], 3)
        self.assertEqual(
            [m["ref_index"] for m in occ["mapping"]], [0, 2, 3]
        )

    def test_same_span_prefers_more_matches(self):
        # 同一首尾与倍率下，匹配拍数多的映射优先（即使误差更大）
        ref = [0.0, 1.0, 2.0, 3.0]
        onsets = [0.0, 1.4, 1.6, 3.0]
        out = _search(ref=ref, onsets=onsets, tolerance=0.5)
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        # 两个中间拍都能匹配（各 0.4 误差）优于只匹配一个
        self.assertEqual(occ["matched_count"], 4)
        self.assertAlmostEqual(occ["total_abs_error"], 0.8, places=6)

    def test_same_span_same_count_prefers_smaller_error(self):
        # 匹配拍数相同，取总绝对误差更小的映射
        ref = [0.0, 1.0, 2.0, 3.0]
        onsets = [0.0, 0.9, 1.2, 3.0]
        out = _search(ref=ref, onsets=onsets, tolerance=0.5)
        occ = out["occurrences"][0]
        self.assertEqual(occ["matched_count"], 3)
        # 第 1 拍应匹配 0.9（误差 0.1）而非 1.2（误差 0.2）
        self.assertEqual(
            [(m["ref_index"], m["onset_index"]) for m in occ["mapping"]],
            [(0, 0), (1, 1), (3, 3)],
        )
        self.assertAlmostEqual(occ["total_abs_error"], 0.1, places=6)

    def test_multiple_ratios_searched_in_order(self):
        # 同一录音在倍率 2.0 下出现；倍率 1.0 下无完整出现
        onsets = [1.0, 2.0, 3.0, 4.0]
        out = _search(onsets=onsets, ratios=(1.0, 2.0))
        self.assertEqual(out["occurrence_count"], 1)
        self.assertEqual(out["occurrences"][0]["ratio"], 2.0)
        self.assertEqual(out["occurrences"][0]["ratio_index"], 1)

    def test_global_selection_maximizes_non_overlapping_count(self):
        # 低速候选 [0,6]（倍率 4，误差 0）与两段常速出现重叠：
        # 两段互不重叠的出现（计数 2）优于单个长候选（计数 1）
        onsets = [0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 4.5, 5.0, 5.5, 6.0]
        out = _search(onsets=onsets, ratios=(4.0, 1.0), tolerance=0.05)
        self.assertEqual(out["occurrence_count"], 2)
        self.assertTrue(all(o["ratio"] == 1.0 for o in out["occurrences"]))
        self.assertEqual(
            [(o["start_time"], o["end_time"]) for o in out["occurrences"]],
            [(0.0, 1.5), (4.0, 5.5)],
        )

    def test_overlapping_long_candidate_loses_on_count(self):
        # 只有长候选覆盖全部起音时它才入选；存在两段短出现时短出现组合优先
        onsets = [0.0, 2.0, 4.0, 6.0]
        out = _search(onsets=onsets, ratios=(4.0,), tolerance=0.05)
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual(occ["ratio"], 4.0)
        self.assertEqual((occ["start_time"], occ["end_time"]), (0.0, 6.0))
        self.assertEqual(occ["matched_count"], 4)

    def test_global_tie_prefers_smaller_total_error(self):
        # 两个重叠候选各算一次出现：保留总绝对误差小者
        ref = [0.0, 1.0, 2.0, 3.0]
        # 候选 A：[0,3]，中间拍 1.1 / 1.9（误差 0.1+0.1=0.2）
        # 候选 B：[0.5,3.5]，中间拍 1.5 / 2.5（误差 0）-> 选 B
        onsets = [0.0, 0.5, 1.1, 1.5, 1.9, 2.5, 3.0, 3.5]
        out = _search(ref=ref, onsets=onsets, tolerance=0.2)
        self.assertEqual(out["occurrence_count"], 1)
        occ = out["occurrences"][0]
        self.assertEqual((occ["start_time"], occ["end_time"]), (0.5, 3.5))
        self.assertAlmostEqual(occ["total_abs_error"], 0.0, places=6)

    def test_global_tie_prefers_earlier_start(self):
        # 计数与总误差都相同的重叠候选：起点早者优先
        ref = [0.0, 1.0, 2.0, 3.0]
        # A: [1,4] 误差 0；B: [0,3] 误差 0，两者重叠 -> 选起点更早的 B
        onsets = [0.0, 1.0, 2.0, 3.0, 4.0]
        out = _search(ref=ref, onsets=onsets, tolerance=0.05)
        self.assertEqual(out["occurrence_count"], 1)
        self.assertEqual(out["occurrences"][0]["start_time"], 0.0)
        self.assertEqual(out["occurrences"][0]["end_time"], 3.0)

    def test_global_tie_prefers_earlier_ratio_input_order(self):
        # 重复倍率产生完全相同的候选：选输入顺序靠前者（ratio_index 小）
        onsets = [0.0, 0.5, 1.0, 1.5]
        out = _search(onsets=onsets, ratios=(1.0, 1.0))
        self.assertEqual(out["occurrence_count"], 1)
        self.assertEqual(out["occurrences"][0]["ratio_index"], 0)

    def test_touching_occurrences_both_selected(self):
        # 前一出现的尾音 = 后一出现的首音（端点相接不算重叠）
        ref = [0.0, 1.0, 2.0]
        onsets = [0.0, 1.0, 2.0, 3.0, 4.0]
        out = _search(ref=ref, onsets=onsets, tolerance=0.05)
        self.assertEqual(out["occurrence_count"], 2)
        self.assertEqual(
            [(o["start_time"], o["end_time"]) for o in out["occurrences"]],
            [(0.0, 2.0), (2.0, 4.0)],
        )

    def test_no_hit_returns_empty_list(self):
        onsets = [10.0, 10.1, 10.2, 10.3]
        out = _search(onsets=onsets, tolerance=0.01)
        self.assertEqual(out["occurrence_count"], 0)
        self.assertEqual(out["occurrences"], [])
        self.assertEqual(out["candidate_count"], 0)

    def test_short_phrase_or_recording_returns_empty(self):
        # 参考拍不足 3 个或起音不足 3 个：不可能满足总匹配 >= 3
        self.assertEqual(
            _search(ref=[0.0, 1.0], onsets=[0.0, 1.0])["occurrences"], []
        )
        self.assertEqual(
            _search(onsets=[0.0, 0.5])["occurrences"], []
        )
        out = _search(ref=[0.0, 1.0], onsets=[0.0, 1.0])
        self.assertEqual(out["occurrence_count"], 0)
        self.assertEqual(out["candidate_count"], 0)

    def test_error_relative_to_first_matched_onset(self):
        # 整体偏移 +0.3 的出现：各拍误差仍相对首匹配起音计算，应为 0
        onsets = [0.3, 0.8, 1.3, 1.8]
        out = _search(onsets=onsets)
        self.assertEqual(out["occurrence_count"], 1)
        self.assertEqual(out["occurrences"][0]["total_abs_error"], 0.0)
        self.assertEqual(out["occurrences"][0]["start_time"], 0.3)

    def test_output_structure_and_parameters(self):
        onsets = [0.0, 0.5, 1.0, 1.5]
        out = _search(onsets=onsets, ratios=(1.0, 2.0), tolerance=0.1)
        self.assertEqual(out["reference_beat_count"], 4)
        self.assertEqual(out["onset_count"], 4)
        self.assertEqual(
            out["parameters"],
            {"tolerance": 0.1, "ratios": [1.0, 2.0], "min_matched_beats": 3},
        )
        occ = out["occurrences"][0]
        self.assertEqual(occ["occurrence_index"], 0)
        for m in occ["mapping"]:
            self.assertIn("ref_time", m)
            self.assertIn("onset_time", m)
            self.assertIn("error", m)


class TestPhraseValidation(unittest.TestCase):
    def test_non_increasing_reference_rejected(self):
        with self.assertRaises(PhraseError):
            _search(ref=[0.0, 0.0, 1.0], onsets=[0.0, 0.5, 1.0])

    def test_non_increasing_onsets_rejected(self):
        with self.assertRaises(PhraseError):
            _search(onsets=[0.0, 0.5, 0.5, 1.0])

    def test_non_positive_ratio_rejected(self):
        with self.assertRaises(PhraseError):
            _search(onsets=[0.0, 0.5, 1.0], ratios=(0.0,))
        with self.assertRaises(PhraseError):
            _search(onsets=[0.0, 0.5, 1.0], ratios=(1.0, -2.0))

    def test_negative_tolerance_rejected(self):
        with self.assertRaises(PhraseError):
            _search(onsets=[0.0, 0.5, 1.0], tolerance=-0.1)

    def test_empty_inputs_rejected(self):
        with self.assertRaises(PhraseError):
            _search(ref=[], onsets=[0.0])
        with self.assertRaises(PhraseError):
            _search(onsets=[])
        with self.assertRaises(PhraseError):
            _search(onsets=[0.0, 0.5, 1.0], ratios=[])


if __name__ == "__main__":
    unittest.main()
