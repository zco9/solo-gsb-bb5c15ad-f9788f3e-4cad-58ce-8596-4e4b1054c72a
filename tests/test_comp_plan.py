"""剪辑方案（compilation plan）纯算法单元测试。"""

import unittest

from app.comp_plan import CompPlanError, build_comp_plan

REF = [0.5 * i for i in range(6)]  # 6 拍，每小节 3 拍 -> 2 个小节


def _job(job_id, dev_by_ref, name=None, ref=None):
    """构造与 crud 行同构的已完成任务：dev_by_ref 为 {参考拍索引: 偏差}。

    未出现在字典中的参考拍视为漏拍；perf_time = ref_time + 偏差。
    """
    grid = list(REF if ref is None else ref)
    matched = [
        {
            "ref_index": ri,
            "perf_index": k,
            "ref_time": grid[ri],
            "perf_time": round(grid[ri] + dev, 6),
            "deviation": round(dev, 6),
            "status": "on_time",
        }
        for k, (ri, dev) in enumerate(sorted(dev_by_ref.items()))
    ]
    return {
        "id": job_id,
        "name": name or f"take-{job_id}",
        "reference_beats": grid,
        "status": "done",
        "result": {"matched_beats": matched, "missed_beats": []},
    }


class TestCompPlan(unittest.TestCase):
    def test_prefers_lower_mad_and_zero_switch(self):
        # 轨1 两小节均匹配：小节 0 偏差 0.01，小节 1 偏差 0.04
        # 轨2 两小节均匹配：小节 0 偏差 0.02，小节 1 偏差 0.02
        jobs = [
            _job(1, {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.04, 4: 0.04, 5: 0.04}),
            _job(2, {0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02}),
        ]
        out = build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.0)
        # 零换轨代价：每小节独立取更优轨 -> 轨1（0.01）、轨2（0.02）
        self.assertEqual([s["job_id"] for s in out["segments"]], [1, 2])
        self.assertEqual(out["switch_count"], 1)
        self.assertAlmostEqual(out["total_cost"], 0.03, places=6)
        self.assertEqual(out["bar_count"], 2)
        self.assertEqual(out["segment_count"], 2)

    def test_high_switch_cost_keeps_single_source(self):
        jobs = [
            _job(1, {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.04, 4: 0.04, 5: 0.04}),
            _job(2, {0: 0.02, 1: 0.02, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02}),
        ]
        # 换轨 0.05 > 两小节代价差（0.05−0.03=0.02）-> 全程轨2
        out = build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.05)
        self.assertEqual(len(out["segments"]), 1)
        seg = out["segments"][0]
        self.assertEqual(seg["job_id"], 2)
        self.assertEqual(seg["bar_start"], 1)
        self.assertEqual(seg["bar_end"], 2)
        self.assertEqual(seg["bar_count"], 2)
        self.assertEqual(seg["ref_start_index"], 0)
        self.assertEqual(seg["ref_end_index"], 5)
        self.assertEqual(seg["beat_number_start"], 1)
        self.assertEqual(seg["beat_number_end"], 6)
        self.assertEqual(seg["beat_count"], 6)
        self.assertEqual(seg["source_start_perf_index"], 0)
        self.assertEqual(seg["source_end_perf_index"], 5)
        self.assertEqual(out["switch_count"], 0)
        self.assertAlmostEqual(out["total_cost"], 0.04, places=6)

    def test_tie_prefers_earlier_request_order(self):
        # 两轨两小节代价完全相同 -> 全程优先请求顺序靠前的轨 1（单片段）
        jobs = [
            _job(1, {ri: 0.01 for ri in range(6)}),
            _job(2, {ri: 0.01 for ri in range(6)}),
        ]
        out = build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.02)
        self.assertEqual([s["job_id"] for s in out["segments"]], [1])
        self.assertEqual(out["switch_count"], 0)
        self.assertAlmostEqual(out["total_cost"], 0.02, places=6)

    def test_tie_switch_cost_boundary_stay_then_switch(self):
        # 小节0：轨1=0.01、轨2=0.03；小节1：轨1=0.04、轨2=0.02
        # 全程轨1 = 0.05；换轨 1->2 = 0.01+0.02+0.02 = 0.05（同代价）
        # 同代价优先请求顺序靠前的任务：首小节取轨1，其后字典序最优续接
        # 也是轨1（[轨1, 轨1] 与 [轨1, 轨2] 同代价时前者更靠前）-> [轨1, 轨1]
        jobs = [
            _job(1, {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.04, 4: 0.04, 5: 0.04}),
            _job(2, {0: 0.03, 1: 0.03, 2: 0.03, 3: 0.02, 4: 0.02, 5: 0.02}),
        ]
        out = build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.02)
        self.assertEqual([s["job_id"] for s in out["segments"]], [1])
        self.assertEqual(out["switch_count"], 0)
        self.assertAlmostEqual(out["total_cost"], 0.05, places=6)

        # 换轨代价再大一点 -> 换轨不再划算，全程轨1
        out = build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.021)
        self.assertEqual([s["job_id"] for s in out["segments"]], [1])
        self.assertAlmostEqual(out["total_cost"], 0.05, places=6)

        # 换轨为零 -> 小节1 直接选更优的轨2
        out = build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.0)
        self.assertEqual([s["job_id"] for s in out["segments"]], [1, 2])
        self.assertEqual(out["switch_count"], 1)
        self.assertAlmostEqual(out["total_cost"], 0.03, places=6)

    def test_incomplete_bar_take_unavailable_forces_switch(self):
        # 轨1 漏小节1 的首拍（索引3），仅可用于小节0
        # 轨2 漏小节0 的首拍（索引0），仅可用于小节1 -> 必须各取一节并换轨
        jobs = [
            _job(1, {0: 0.01, 1: 0.01, 2: 0.01, 4: 0.01, 5: 0.01}),
            _job(2, {1: 0.0, 2: 0.0, 3: 0.02, 4: 0.02, 5: 0.02}),
        ]
        out = build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.05)
        self.assertEqual([s["job_id"] for s in out["segments"]], [1, 2])
        self.assertEqual(out["switch_count"], 1)
        # 小节选用代价 0.01 + 0.02，外加一次换轨 0.05
        self.assertAlmostEqual(out["total_cost"], 0.08, places=6)
        first, second = out["segments"]
        # 来源首末匹配音时间
        self.assertAlmostEqual(first["source_start_time"], REF[0] + 0.01, places=6)
        self.assertAlmostEqual(first["source_end_time"], REF[2] + 0.01, places=6)
        self.assertAlmostEqual(second["source_start_time"], REF[3] + 0.02, places=6)
        self.assertAlmostEqual(second["source_end_time"], REF[5] + 0.02, places=6)
        self.assertEqual(first["source_start_perf_index"], 0)
        self.assertEqual(first["source_end_perf_index"], 2)
        # 第二轨小节1 从其匹配序列的第 2 个音（漏过索引0）开始
        self.assertEqual(second["source_start_perf_index"], 2)
        self.assertEqual(second["source_end_perf_index"], 4)
        self.assertAlmostEqual(first["ref_start_time"], 0.0, places=6)
        self.assertAlmostEqual(second["ref_end_time"], 2.5, places=6)

    def test_no_take_covers_bar_raises_with_bar_number(self):
        # 小节1（索引 3..5）两轨都漏了索引 3
        jobs = [
            _job(1, {0: 0.0, 1: 0.0, 2: 0.0, 4: 0.0, 5: 0.0}),
            _job(2, {0: 0.0, 1: 0.0, 2: 0.0, 4: 0.0}),
        ]
        with self.assertRaises(CompPlanError) as ctx:
            build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.0)
        self.assertIn("2", str(ctx.exception))
        self.assertIn("小节", str(ctx.exception))

    def test_multiple_unavailable_bars_listed(self):
        # 每小节 1 拍；索引 1 与 3 两轨都漏 -> 第 2、4 小节不可用
        jobs = [
            _job(1, {0: 0.0, 2: 0.0, 4: 0.0, 5: 0.0}),
            _job(2, {0: 0.0, 2: 0.0, 5: 0.0}),
        ]
        with self.assertRaises(CompPlanError) as ctx:
            build_comp_plan(jobs, beats_per_bar=1, switch_cost=0.0)
        message = str(ctx.exception)
        self.assertIn("2", message)
        self.assertIn("4", message)

    def test_trailing_partial_bar_is_its_own_bar(self):
        # 7 拍、每小节 3 拍 -> 3 个小节（末尾小节仅 1 拍）
        ref = [0.5 * i for i in range(7)]
        jobs = [
            _job(1, {ri: 0.01 for ri in range(7)}, ref=ref),
            _job(2, {ri: 0.02 for ri in range(7)}, ref=ref),
        ]
        out = build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.0)
        self.assertEqual(out["bar_count"], 3)
        # 三小节都选轨1；末尾片段只覆盖第 7 拍
        self.assertEqual([s["job_id"] for s in out["segments"]], [1])
        seg = out["segments"][0]
        self.assertEqual(seg["bar_start"], 1)
        self.assertEqual(seg["bar_end"], 3)
        self.assertEqual(seg["ref_start_index"], 0)
        self.assertEqual(seg["ref_end_index"], 6)
        # 末尾单拍小节的匹配拍平均绝对偏差仍计入总代价
        self.assertAlmostEqual(out["total_cost"], 0.03, places=6)

    def test_merged_segment_costs_and_mad(self):
        jobs = [
            # 轨1 小节0 偏差 0.01；轨2 小节1 偏差 0.03
            _job(1, {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.09, 4: 0.09, 5: 0.09}),
            _job(2, {0: 0.08, 1: 0.08, 2: 0.08, 3: 0.03, 4: 0.03, 5: 0.03}),
        ]
        out = build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.0)
        first, second = out["segments"]
        self.assertAlmostEqual(first["selection_cost"], 0.01, places=6)
        self.assertAlmostEqual(first["mean_absolute_deviation"], 0.01, places=6)
        self.assertAlmostEqual(second["selection_cost"], 0.03, places=6)
        self.assertAlmostEqual(second["mean_absolute_deviation"], 0.03, places=6)
        self.assertEqual(first["name"], "take-1")

    def test_segment_merges_consecutive_same_source_bars(self):
        # 3 个小节：最优序列 [轨1, 轨2, 轨2]（switch=0 时逐节取更优）
        # -> 轨2 的两个连续小节必须合并成一个片段
        jobs = [
            _job(1, {
                0: 0.01, 1: 0.01, 2: 0.09, 3: 0.09, 4: 0.03, 5: 0.03,
            }),
            _job(2, {
                0: 0.08, 1: 0.08, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02,
            }),
        ]
        out = build_comp_plan(jobs, beats_per_bar=2, switch_cost=0.0)
        self.assertEqual(out["bar_count"], 3)
        self.assertEqual([s["job_id"] for s in out["segments"]], [1, 2])
        second = out["segments"][1]
        self.assertEqual(second["bar_start"], 2)
        self.assertEqual(second["bar_end"], 3)
        self.assertEqual(second["bar_count"], 2)
        self.assertEqual(second["ref_start_index"], 2)
        self.assertEqual(second["ref_end_index"], 5)
        self.assertEqual(second["beat_count"], 4)
        self.assertAlmostEqual(second["selection_cost"], 0.04, places=6)
        self.assertAlmostEqual(
            second["mean_absolute_deviation"], 0.02, places=6
        )

    def test_validation_errors(self):
        good = [
            _job(1, {ri: 0.01 for ri in range(6)}),
            _job(2, {ri: 0.01 for ri in range(6)}),
        ]
        with self.assertRaises(CompPlanError):
            build_comp_plan(good, beats_per_bar=0, switch_cost=0.0)
        with self.assertRaises(CompPlanError):
            build_comp_plan(good, beats_per_bar=3, switch_cost=-0.01)
        with self.assertRaises(CompPlanError):
            build_comp_plan([good[0]], beats_per_bar=3, switch_cost=0.0)

    def test_job_ids_follow_request_order(self):
        jobs = [
            _job(11, {ri: 0.01 for ri in range(6)}),
            _job(22, {ri: 0.01 for ri in range(6)}),
        ]
        out = build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.0)
        self.assertEqual(out["job_ids"], [11, 22])
        self.assertEqual(out["take_count"], 2)
        self.assertEqual(out["reference_beat_count"], 6)
        self.assertEqual(out["beats_per_bar"], 3)
        self.assertEqual(out["parameters"], {"beats_per_bar": 3, "switch_cost": 0.0})

    def test_signed_deviations_use_absolute_value(self):
        # 抢拍（负偏差）与拖拍（正偏差）按绝对值计入选用代价
        jobs = [
            _job(1, {0: -0.02, 1: 0.02, 2: -0.02, 3: 0.0, 4: 0.0, 5: 0.0}),
            _job(2, {ri: 0.01 for ri in range(6)}),
        ]
        out = build_comp_plan(jobs, beats_per_bar=3, switch_cost=0.0)
        # 小节0：轨1 MAD=0.02 劣于轨2 0.01；小节1：轨1 MAD=0 优于轨2
        self.assertEqual([s["job_id"] for s in out["segments"]], [2, 1])


if __name__ == "__main__":
    unittest.main()
