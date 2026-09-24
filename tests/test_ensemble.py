"""合奏一致性（ensemble consensus）纯算法单元测试。"""

import unittest

from app.ensemble import EnsembleError, build_consensus

REF = [0.0, 0.5, 1.0, 1.5]


def _job(job_id, delay, matched_refs, missed_refs=(), name=None):
    """构造与 crud 行同构的任务字典：matched_refs 为匹配参考拍索引。

    每条匹配拍的 perf_time = ref_time + delay（恒定延迟，无演奏偏差）；
    漏拍由 missed_refs 给出。
    """
    matched = [
        {
            "ref_index": ri,
            "perf_index": k,
            "ref_time": REF[ri],
            "perf_time": REF[ri] + delay,
            "deviation": round(delay, 6),
            "status": "on_time",
        }
        for k, ri in enumerate(matched_refs)
    ]
    return {
        "id": job_id,
        "name": name or f"track-{job_id}",
        "reference_beats": list(REF),
        "status": "done",
        "result": {
            "matched_beats": matched,
            "missed_beats": [
                {"ref_index": ri, "ref_time": REF[ri]} for ri in missed_refs
            ],
        },
    }


class TestEnsembleConsensus(unittest.TestCase):
    def test_constant_delays_corrected_to_grid(self):
        # 三轨固定延迟 0.1 / 0.2 / 0.3，全部匹配：校正后全部落在网格上
        jobs = [
            _job(1, 0.1, range(4)),
            _job(2, 0.2, range(4)),
            _job(3, 0.3, range(4)),
        ]
        out = build_consensus(jobs, 0.05)
        self.assertEqual([t["delay"] for t in out["tracks"]], [0.1, 0.2, 0.3])
        self.assertEqual(out["decidable_beat_count"], 4)
        self.assertEqual(out["undecidable_beat_count"], 0)
        self.assertEqual(out["outlier_count"], 0)
        for beat in out["beats"]:
            self.assertTrue(beat["decidable"])
            self.assertEqual(beat["matched_track_count"], 3)
            self.assertAlmostEqual(
                beat["consensus_time"], beat["ref_time"], places=6
            )
            self.assertEqual(beat["outliers"], [])
            self.assertEqual(beat["missing_job_ids"], [])
            self.assertEqual(sorted(beat["matched_job_ids"]), [1, 2, 3])

    def test_median_delay_robust_to_one_outlier_beat(self):
        # 轨1 延迟 0.1，但最后一拍演奏偏差 +0.09（拖拍）；
        # 延迟中位数仍为 0.1，异常留在该拍
        jobs = [
            _job(1, 0.1, range(4)),
            _job(2, 0.1, range(4)),
            _job(3, 0.1, range(4)),
        ]
        jobs[0]["result"]["matched_beats"][3]["perf_time"] = REF[3] + 0.19
        jobs[0]["result"]["matched_beats"][3]["deviation"] = 0.19
        out = build_consensus(jobs, 0.05)
        self.assertEqual(out["tracks"][0]["delay"], 0.1)
        last = out["beats"][3]
        self.assertEqual(last["outlier_count"], 1)
        outlier = last["outliers"][0]
        self.assertEqual(outlier["job_id"], 1)
        self.assertEqual(outlier["status"], "late")
        self.assertAlmostEqual(outlier["deviation"], 0.09, places=6)
        # 前三拍无异常
        self.assertEqual(sum(b["outlier_count"] for b in out["beats"][:3]), 0)

    def test_early_outlier_sign(self):
        jobs = [
            _job(1, 0.0, range(4)),
            _job(2, 0.0, range(4)),
            _job(3, 0.0, range(4)),
        ]
        # 轨3 第二拍抢拍 0.08
        jobs[2]["result"]["matched_beats"][1]["perf_time"] = REF[1] - 0.08
        jobs[2]["result"]["matched_beats"][1]["deviation"] = -0.08
        out = build_consensus(jobs, 0.05)
        beat = out["beats"][1]
        self.assertEqual(beat["outlier_count"], 1)
        self.assertEqual(beat["outliers"][0]["job_id"], 3)
        self.assertEqual(beat["outliers"][0]["status"], "early")
        self.assertAlmostEqual(beat["outliers"][0]["deviation"], -0.08, places=6)

    def test_threshold_boundary_strict(self):
        jobs = [
            _job(1, 0.0, range(4)),
            _job(2, 0.0, range(4)),
            _job(3, 0.0, range(4)),
        ]
        # 偏差恰好等于阈值 0.05 -> 严格大于才报异常，不报
        jobs[2]["result"]["matched_beats"][0]["perf_time"] = REF[0] + 0.05
        out = build_consensus(jobs, 0.05)
        self.assertEqual(out["beats"][0]["outlier_count"], 0)

    def test_even_track_count_baseline_is_average(self):
        # 4 轨延迟均为 0；首拍校正时间 0.0 / 0.0 / 0.02 / 0.02
        # -> 偶数个值的中位数取中间均值 0.01，均未超阈值
        jobs = [
            _job(1, 0.0, range(4)),
            _job(2, 0.0, range(4)),
            _job(3, 0.0, range(4)),
            _job(4, 0.0, range(4)),
        ]
        jobs[2]["result"]["matched_beats"][0]["perf_time"] = 0.02
        jobs[3]["result"]["matched_beats"][0]["perf_time"] = 0.02
        out = build_consensus(jobs, 0.05)
        first = out["beats"][0]
        self.assertEqual(first["matched_track_count"], 4)
        self.assertAlmostEqual(first["consensus_time"], 0.01, places=6)
        self.assertEqual(first["outlier_count"], 0)

    def test_undecidable_when_only_two_tracks_match(self):
        # 4 轨，最后一拍仅轨 1、2 匹配 -> 不可判定、无基准、无异常
        jobs = [
            _job(1, 0.0, range(4)),
            _job(2, 0.0, range(4)),
            _job(3, 0.0, range(3), missed_refs=(3,)),
            _job(4, 0.0, range(3), missed_refs=(3,)),
        ]
        out = build_consensus(jobs, 0.05)
        self.assertEqual(out["decidable_beat_count"], 3)
        self.assertEqual(out["undecidable_beat_count"], 1)
        last = out["beats"][3]
        self.assertFalse(last["decidable"])
        self.assertIsNone(last["consensus_time"])
        self.assertEqual(last["matched_track_count"], 2)
        self.assertEqual(sorted(last["matched_job_ids"]), [1, 2])
        self.assertEqual(sorted(last["missing_job_ids"]), [3, 4])
        self.assertEqual(last["outliers"], [])
        self.assertEqual(last["outlier_count"], 0)

        # 漏拍逐轨列出
        missed_3 = [b["ref_index"] for b in out["tracks"][2]["missed_beats"]]
        missed_4 = [b["ref_index"] for b in out["tracks"][3]["missed_beats"]]
        self.assertEqual(missed_3, [3])
        self.assertEqual(missed_4, [3])
        self.assertEqual(out["tracks"][0]["missed_beats"], [])
        self.assertEqual(out["tracks"][2]["missed_beats"][0]["beat_number"], 4)

    def test_track_without_matched_beats_delay_none(self):
        # 构造一条没有任何匹配拍的轨（result 中 matched_beats 为空）
        empty = _job(9, 0.0, (), missed_refs=range(4))
        jobs = [_job(1, 0.0, range(4)), _job(2, 0.0, range(4)), empty]
        out = build_consensus(jobs, 0.05)
        self.assertIsNone(out["tracks"][2]["delay"])
        # 每拍只有 2 轨匹配 -> 全部不可判定
        self.assertEqual(out["decidable_beat_count"], 0)
        self.assertEqual(out["undecidable_beat_count"], 4)

    def test_fewer_than_three_jobs_rejected(self):
        with self.assertRaises(EnsembleError):
            build_consensus([_job(1, 0.0, range(4))], 0.05)

    def test_non_positive_threshold_rejected(self):
        jobs = [
            _job(1, 0.0, range(4)),
            _job(2, 0.0, range(4)),
            _job(3, 0.0, range(4)),
        ]
        with self.assertRaises(EnsembleError):
            build_consensus(jobs, 0.0)
        with self.assertRaises(EnsembleError):
            build_consensus(jobs, -0.1)

    def test_two_tracks_deviating_cancel_out_as_median_pair(self):
        # 3 轨延迟均为 0（多数拍落在网格上）；第 3 拍轨1 抢 0.2、轨3 拖 0.2
        # -> 校正时间 {-0.2, 0, +0.2} 中位数为 0，两轨都超阈值
        jobs = [
            _job(1, 0.0, range(4)),
            _job(2, 0.0, range(4)),
            _job(3, 0.0, range(4)),
        ]
        jobs[0]["result"]["matched_beats"][2]["perf_time"] = REF[2] - 0.2
        jobs[2]["result"]["matched_beats"][2]["perf_time"] = REF[2] + 0.2
        out = build_consensus(jobs, 0.05)
        beat = out["beats"][2]
        self.assertEqual(beat["outlier_count"], 2)
        statuses = {o["job_id"]: o["status"] for o in beat["outliers"]}
        self.assertEqual(statuses[1], "early")
        self.assertEqual(statuses[3], "late")


if __name__ == "__main__":
    unittest.main()
