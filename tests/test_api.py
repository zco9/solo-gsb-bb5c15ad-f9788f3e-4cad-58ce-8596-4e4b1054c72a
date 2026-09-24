"""FastAPI 接口集成测试：使用临时 SQLite 文件。"""

import os
import tempfile
import unittest

# 必须在导入 app.* 之前设置数据库路径（app.database 在导入时读取该变量）
_TMPDIR = tempfile.mkdtemp(prefix="beat-align-test-")
os.environ["ALIGN_DB_PATH"] = os.path.join(_TMPDIR, "test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

REF = [0.0, 0.5, 1.0, 1.5, 2.0]
PERF = [0.0, 0.48, 1.52, 2.0]  # 漏第 3 拍(1.0)，且有轻微抢/拖


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cm = TestClient(app)
        cls.client = cls._cm.__enter__()
        cls.addClassCleanup(cls._cm.__exit__, None, None, None)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_full_flow(self):
        # 1. 创建
        r = self.client.post("/jobs", json={
            "name": "take-01",
            "reference_beats": REF,
            "performance_onsets": PERF,
            "params": {"band_width": 2},
        })
        self.assertEqual(r.status_code, 201, r.text)
        job = r.json()
        self.assertEqual(job["status"], "pending")
        job_id = job["id"]

        # 2. 对齐
        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200, r.text)
        done = r.json()
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["result"]["statistics"]["missed_count"], 1)
        self.assertEqual(done["result"]["statistics"]["matched_count"], 4)

        # 3. 查询
        r = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "done")

        r = self.client.get(f"/jobs/{job_id}/result")
        self.assertEqual(r.status_code, 200)
        self.assertIn("matched_beats", r.json()["result"])

        r = self.client.get("/jobs")
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(len(r.json()), 1)

        # 4. 重复对齐不重算；force=true 可重算
        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200)
        r = self.client.post(f"/jobs/{job_id}/align?force=true")
        self.assertEqual(r.status_code, 200)

    def test_404(self):
        r = self.client.get("/jobs/999999")
        self.assertEqual(r.status_code, 404)
        r = self.client.post("/jobs/999999/align")
        self.assertEqual(r.status_code, 404)

    def test_result_before_align_conflicts(self):
        r = self.client.post("/jobs", json={
            "reference_beats": REF, "performance_onsets": REF,
        })
        job_id = r.json()["id"]
        r = self.client.get(f"/jobs/{job_id}/result")
        self.assertEqual(r.status_code, 409)

    def test_band_too_narrow_rejected(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0, 1.5],
            "performance_onsets": [0.0],
            "params": {"band_width": 1},
        })
        self.assertEqual(r.status_code, 400)

    def test_non_monotonic_rejected(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.6, 0.5],
            "performance_onsets": [0.0, 0.5],
        })
        self.assertEqual(r.status_code, 422)

    def test_anchor_full_flow(self):
        # 自由对齐会漏掉 ref[1]、ref[2]；锚点强制 (1,1)、(2,2) 匹配
        body = {
            "name": "anchor-take",
            "reference_beats": [0.0, 1.0, 2.0, 3.0],
            "performance_onsets": [0.0, 0.1, 2.9],
            "params": {"band_width": 3},
            "anchors": [
                {"ref_index": 1, "perf_index": 1},
                {"ref_index": 2, "perf_index": 2},
            ],
        }
        r = self.client.post("/jobs", json=body)
        self.assertEqual(r.status_code, 201, r.text)
        job = r.json()
        # 任务查询包含提交的锚点（SQLite 持久化往返）
        self.assertEqual(job["anchors"], body["anchors"])
        job_id = job["id"]

        r = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["anchors"], body["anchors"])

        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200, r.text)
        result = r.json()["result"]
        pairs = [(b["ref_index"], b["perf_index"]) for b in result["matched_beats"]]
        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 2)])
        # result 中包含已采用锚点
        self.assertEqual(
            [(a["ref_index"], a["perf_index"]) for a in result["anchors"]],
            [(1, 1), (2, 2)],
        )
        self.assertEqual(result["anchors"][0]["deviation"], -0.9)

        # /result 端点同样可查到已采用锚点
        r = self.client.get(f"/jobs/{job_id}/result")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["result"]["anchors"]), 2)

        # 重新查询确认结果已从 SQLite 读出
        r = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(len(r.json()["result"]["anchors"]), 2)

    def test_anchor_skipped_within_segment(self):
        # 锚点之间仍允许漏拍
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0, 1.5, 2.0],
            "performance_onsets": [0.0, 0.5, 1.5, 2.0],
            "params": {"band_width": 3},
            "anchors": [{"ref_index": 0, "perf_index": 0},
                        {"ref_index": 4, "perf_index": 3}],
        })
        self.assertEqual(r.status_code, 201, r.text)
        job_id = r.json()["id"]
        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200, r.text)
        result = r.json()["result"]
        self.assertEqual([b["ref_index"] for b in result["missed_beats"]], [2])

    def test_anchor_out_of_range_400(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0],
            "performance_onsets": [0.0, 0.5, 1.0],
            "params": {"band_width": 1},
            "anchors": [{"ref_index": 3, "perf_index": 0}],
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("越界", r.json()["detail"])

    def test_anchor_not_increasing_400(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0, 1.5],
            "performance_onsets": [0.0, 0.5, 1.0, 1.5],
            "params": {"band_width": 2},
            "anchors": [{"ref_index": 1, "perf_index": 1},
                        {"ref_index": 1, "perf_index": 2}],
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("严格递增", r.json()["detail"])

    def test_anchor_band_unreachable_400(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0, 1.5],
            "performance_onsets": [0.0, 0.5, 1.0, 1.5],
            "params": {"band_width": 1},
            "anchors": [{"ref_index": 3, "perf_index": 0}],
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("带宽", r.json()["detail"])

    def test_anchor_negative_index_400(self):
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0],
            "performance_onsets": [0.0, 0.5, 1.0],
            "anchors": [{"ref_index": -1, "perf_index": 0}],
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("不能为负", r.json()["detail"])

    def test_job_without_anchors_compatible(self):
        # 未提供 anchors 的任务：输入侧返回 []，结果侧返回 []
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 0.5, 1.0],
            "performance_onsets": [0.0, 0.5, 1.0],
        })
        self.assertEqual(r.status_code, 201, r.text)
        job_id = r.json()["id"]
        self.assertEqual(r.json()["anchors"], [])
        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["result"]["anchors"], [])

    def test_legacy_db_without_anchors_column(self):
        # 模拟旧版数据库（jobs 表没有 anchors 列）：迁移后旧行可读、新任务可建
        import sqlite3
        from app import database
        db_path = os.environ["ALIGN_DB_PATH"]

        # 测试库：丢弃当前表并按旧结构重建，写入一行旧数据
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DROP TABLE IF EXISTS jobs")
            conn.commit()
        finally:
            conn.close()
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    reference_beats TEXT NOT NULL,
                    performance_onsets TEXT NOT NULL,
                    params TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    result TEXT,
                    created_at TEXT NOT NULL,
                    aligned_at TEXT
                );
                INSERT INTO jobs (name, reference_beats, performance_onsets,
                                  params, status, created_at)
                VALUES ('old', '[0, 1]', '[0, 1]', '{}', 'pending',
                        '2026-01-01T00:00:00+00:00');
                """
            )
            conn.commit()
        finally:
            conn.close()

        database.init_db()  # 触发幂等迁移
        # 旧行可读，anchors 兜底为空列表
        r = self.client.get("/jobs")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(any(j["name"] == "old" for j in r.json()))

        # 迁移后仍可正常创建带 / 不带锚点的新任务
        r = self.client.post("/jobs", json={
            "reference_beats": [0.0, 1.0],
            "performance_onsets": [0.0, 1.0],
            "anchors": [{"ref_index": 0, "perf_index": 0}],
        })
        self.assertEqual(r.status_code, 201, r.text)
        self.assertEqual(r.json()["anchors"], [{"ref_index": 0, "perf_index": 0}])

    # ---- 节奏曲线导出 GET /jobs/{id}/tempo-map ----

    def _create_done_job(self, body):
        r = self.client.post("/jobs", json=body)
        self.assertEqual(r.status_code, 201, r.text)
        job_id = r.json()["id"]
        r = self.client.post(f"/jobs/{job_id}/align")
        self.assertEqual(r.status_code, 200, r.text)
        return job_id

    def test_tempo_map_constant_tempo(self):
        job_id = self._create_done_job({
            "reference_beats": [0.5 * i for i in range(8)],
            "performance_onsets": [0.5 * i for i in range(8)],
        })
        r = self.client.get(f"/jobs/{job_id}/tempo-map")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["job_id"], job_id)
        self.assertEqual(data["matched_beat_count"], 8)
        self.assertEqual(data["ignored_ref_indices"], [])
        self.assertEqual(len(data["events"]), 1)
        ev = data["events"][0]
        self.assertAlmostEqual(ev["bpm"], 120.0, places=2)
        self.assertEqual(ev["ref_start_index"], 0)
        self.assertEqual(ev["ref_end_index"], 7)
        self.assertEqual(ev["beat_count"], 7)
        self.assertEqual(ev["gap_count"], 7)
        self.assertEqual(data["parameters"]["window_beats"], 5)

    def test_tempo_map_change_detected_and_params_passthrough(self):
        # 前半 120 BPM、后半 100 BPM；首尾锚点保证 9 拍全部一一匹配
        body = {
            "reference_beats": [0.5 * i for i in range(9)],
            "performance_onsets": [0, .5, 1.0, 1.5, 2.0, 2.6, 3.2, 3.8, 4.4],
            "params": {"band_width": 3},
            "anchors": [{"ref_index": 0, "perf_index": 0},
                        {"ref_index": 8, "perf_index": 8}],
        }
        job_id = self._create_done_job(body)
        r = self.client.get(
            f"/jobs/{job_id}/tempo-map?window_beats=3&min_bpm=50&max_bpm=250"
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["ignored_ref_indices"], [])
        self.assertEqual(len(data["events"]), 2)
        self.assertAlmostEqual(data["events"][0]["bpm"], 120.0, places=1)
        self.assertAlmostEqual(data["events"][1]["bpm"], 100.0, places=1)
        self.assertEqual(data["events"][0]["ref_end_index"],
                         data["events"][1]["ref_start_index"])
        self.assertEqual(data["parameters"]["window_beats"], 3)
        self.assertEqual(data["parameters"]["min_bpm"], 50.0)
        self.assertEqual(data["parameters"]["max_bpm"], 250.0)

    def test_tempo_map_skipped_beat_span_normalized(self):
        # 漏 ref[1]（锚点强制），其余 120 BPM；间隔 0↔2 跨两拍 / 1.0s
        # -> 按参考拍跨度换算仍为 120 BPM，覆盖拍数含被漏的拍
        body = {
            "reference_beats": [0.0, 0.5, 1.0, 1.5, 2.0],
            "performance_onsets": [0.0, 1.0, 1.5, 2.0],
            "params": {"band_width": 3},
            "anchors": [{"ref_index": 0, "perf_index": 0},
                        {"ref_index": 4, "perf_index": 3}],
        }
        job_id = self._create_done_job(body)
        r = self.client.get(f"/jobs/{job_id}/tempo-map?window_beats=3")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["ignored_ref_indices"], [])
        self.assertEqual(len(data["events"]), 1)
        self.assertAlmostEqual(data["events"][0]["bpm"], 120.0, places=1)
        self.assertEqual(data["events"][0]["beat_count"], 4)

    def test_tempo_map_ignored_residual_anomaly(self):
        # 手工持久化一个含残留异常间隔的 done 结果：ref2→ref3 只有 0.15s
        # -> 400 BPM，被中位数窗口与 [40,300] 范围联合判为异常并忽略
        job_id = self._create_done_job({
            "reference_beats": [0.5 * i for i in range(7)],
            "performance_onsets": [0.5 * i for i in range(7)],
        })
        from app import crud, database
        conn = database.connect()
        try:
            job = crud.get_job(conn, job_id)
            result = job["result"]
            result["matched_beats"] = [
                {"ref_index": 0, "perf_index": 0, "ref_time": 0.0,
                 "perf_time": 0.0, "deviation": 0.0, "status": "on_time"},
                {"ref_index": 1, "perf_index": 1, "ref_time": 0.5,
                 "perf_time": 0.5, "deviation": 0.0, "status": "on_time"},
                {"ref_index": 2, "perf_index": 2, "ref_time": 1.0,
                 "perf_time": 1.0, "deviation": 0.0, "status": "on_time"},
                {"ref_index": 3, "perf_index": 3, "ref_time": 1.5,
                 "perf_time": 1.15, "deviation": -0.35, "status": "early"},
                {"ref_index": 4, "perf_index": 4, "ref_time": 2.0,
                 "perf_time": 1.65, "deviation": -0.35, "status": "early"},
                {"ref_index": 5, "perf_index": 5, "ref_time": 2.5,
                 "perf_time": 2.15, "deviation": -0.35, "status": "early"},
                {"ref_index": 6, "perf_index": 6, "ref_time": 3.0,
                 "perf_time": 2.65, "deviation": -0.35, "status": "early"},
            ]
            crud.save_result(conn, job_id, result)
        finally:
            conn.close()

        r = self.client.get(f"/jobs/{job_id}/tempo-map?window_beats=3")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertIn(2, data["ignored_ref_indices"])
        # 异常间隔不切断事件：前后同速，合并为单个 120 BPM 事件
        self.assertEqual(len(data["events"]), 1)
        self.assertAlmostEqual(data["events"][0]["bpm"], 120.0, places=1)
        self.assertEqual(data["events"][0]["gap_count"], 5)

    def test_tempo_map_error_states(self):
        # 404：任务不存在
        r = self.client.get("/jobs/999999/tempo-map")
        self.assertEqual(r.status_code, 404)

        # 409：任务未对齐
        r = self.client.post("/jobs", json={
            "reference_beats": REF, "performance_onsets": PERF,
        })
        pending_id = r.json()["id"]
        r = self.client.get(f"/jobs/{pending_id}/tempo-map")
        self.assertEqual(r.status_code, 409)

        # 400：上次对齐失败
        from app import crud, database
        conn = database.connect()
        try:
            crud.save_error(conn, pending_id, "boom")
        finally:
            conn.close()
        r = self.client.get(f"/jobs/{pending_id}/tempo-map")
        self.assertEqual(r.status_code, 400)
        self.assertIn("boom", r.json()["detail"])

        # 409：已完成但没有匹配拍（手工持久化空 matched_beats）
        empty_id = self._create_done_job({
            "reference_beats": [0.0, 1.0],
            "performance_onsets": [0.0, 1.0],
        })
        conn = database.connect()
        try:
            job = crud.get_job(conn, empty_id)
            result = job["result"]
            result["matched_beats"] = []
            crud.save_result(conn, empty_id, result)
        finally:
            conn.close()
        r = self.client.get(f"/jobs/{empty_id}/tempo-map")
        self.assertEqual(r.status_code, 409)
        self.assertIn("匹配拍", r.json()["detail"])

    def test_tempo_map_param_validation(self):
        job_id = self._create_done_job({
            "reference_beats": [0.5 * i for i in range(6)],
            "performance_onsets": [0.5 * i for i in range(6)],
        })
        # min_bpm >= max_bpm -> 422
        r = self.client.get(f"/jobs/{job_id}/tempo-map?min_bpm=200&max_bpm=100")
        self.assertEqual(r.status_code, 422)
        self.assertIn("min_bpm", r.json()["detail"])
        # window_beats 越界 -> 422
        r = self.client.get(f"/jobs/{job_id}/tempo-map?window_beats=0")
        self.assertEqual(r.status_code, 422)
        r = self.client.get(f"/jobs/{job_id}/tempo-map?window_beats=102")
        self.assertEqual(r.status_code, 422)

    # ---- 复核片段 GET /jobs/{id}/review-segments ----

    def test_review_segments_split_and_merge_by_tolerance(self):
        # 10 个参考拍、8 个演奏音：演奏缺失参考第 1 拍（索引0）与
        # 第 8 拍（索引7），中间锚点 (6,5) 固定错位，DTW 恰好在两处漏拍
        ref = [0.5 * i for i in range(10)]
        perf = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5]
        body = {
            "reference_beats": ref,
            "performance_onsets": perf,
            "params": {"band_width": 3},
            "anchors": [{"ref_index": 6, "perf_index": 5}],
        }
        job_id = self._create_done_job(body)
        r = self.client.get(f"/jobs/{job_id}/result")
        result = r.json()["result"]
        self.assertEqual(
            [b["ref_index"] for b in result["missed_beats"]], [0, 7]
        )
        self.assertEqual(result["inserted_beats"], [])

        # 默认容差 2 -> 两个片段，拍号 1 与 8
        r = self.client.get(f"/jobs/{job_id}/review-segments")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["job_id"], job_id)
        self.assertEqual(data["segment_count"], 2)
        self.assertEqual(data["total_missed_count"], 2)
        self.assertEqual(data["total_inserted_count"], 0)
        self.assertEqual(data["total_over_threshold_count"], 0)
        ranges = [
            (s["beat_number_start"], s["beat_number_end"])
            for s in data["segments"]
        ]
        self.assertEqual(ranges, [(1, 1), (8, 8)])
        seg = data["segments"][0]
        self.assertEqual(seg["issues"][0]["type"], "missed")
        self.assertEqual(seg["issues"][0]["beat_number"], 1)
        self.assertIn("漏拍", seg["summary"])
        self.assertEqual(data["parameters"]["deviation_threshold"], 0.05)
        self.assertEqual(data["parameters"]["gap_tolerance_beats"], 2.0)

        # 容差放大到 7（边缘差正好 7）-> 合并为一个片段，拍号 1–8
        r = self.client.get(
            f"/jobs/{job_id}/review-segments?gap_tolerance_beats=7"
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["segment_count"], 1)
        self.assertEqual(
            (data["segments"][0]["beat_number_start"],
             data["segments"][0]["beat_number_end"]),
            (1, 8),
        )
        self.assertEqual(data["segments"][0]["missed_count"], 2)
        # 容差 6（边缘差 7 > 6）仍分裂
        r = self.client.get(
            f"/jobs/{job_id}/review-segments?gap_tolerance_beats=6"
        )
        self.assertEqual(r.json()["segment_count"], 2)

    def test_review_segments_insert_and_over_threshold(self):
        # 9 参考拍 / 10 演奏音：p1=0.22 是 ref0/ref1 之间的插入音；
        # ref3..ref6 连续拖拍 0.06s；插入音跨度 0..1 与拖拍簇 3..6 间隔 2 拍
        ref = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        perf = [0.0, 0.22, 0.5, 1.0, 1.56, 2.06, 2.56, 3.06, 3.5, 4.0]
        body = {
            "reference_beats": ref,
            "performance_onsets": perf,
            "params": {
                "band_width": 2,
                "skip_penalty": 0.02,
                "insert_penalty": 0.02,
                "on_time_threshold": 0.03,
            },
            "anchors": [{"ref_index": 0, "perf_index": 0},
                        {"ref_index": 8, "perf_index": 9}],
        }
        job_id = self._create_done_job(body)

        # 先核对 DTW 结果：1 个插入音、无漏拍、ref3..6 拖拍
        result = self.client.get(f"/jobs/{job_id}/result").json()["result"]
        self.assertEqual(
            [b["perf_index"] for b in result["inserted_beats"]], [1]
        )
        self.assertEqual(result["missed_beats"], [])

        r = self.client.get(
            f"/jobs/{job_id}/review-segments"
            "?deviation_threshold=0.05&gap_tolerance_beats=2"
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        # 插入音（跨度 0..1）与拖拍簇（3..6）边缘差 2 ≤ 2 -> 一个片段
        self.assertEqual(data["segment_count"], 1)
        seg = data["segments"][0]
        self.assertEqual(
            (seg["beat_number_start"], seg["beat_number_end"]), (1, 7)
        )
        self.assertEqual(seg["inserted_count"], 1)
        self.assertEqual(seg["missed_count"], 0)
        self.assertEqual(seg["over_threshold_count"], 4)
        self.assertEqual(seg["late_count"], 4)
        self.assertEqual(seg["issue_count"], 5)
        self.assertEqual(seg["issues"][0]["type"], "inserted")
        self.assertEqual(seg["issues"][0]["perf_index"], 1)
        self.assertIn("第 1–2 拍", seg["issues"][0]["description"])
        self.assertAlmostEqual(seg["max_abs_deviation"], 0.06, places=6)
        self.assertIn("插入音 1 处", seg["summary"])
        self.assertIn("拖拍 4", seg["summary"])
        self.assertEqual(data["total_inserted_count"], 1)
        self.assertEqual(data["total_over_threshold_count"], 4)

        # 收紧阈值 -> 拖拍不再超阈值，只剩插入音片段（跨度 0..1，拍号 1–2）
        r = self.client.get(
            f"/jobs/{job_id}/review-segments?deviation_threshold=0.07"
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["segment_count"], 1)
        seg = data["segments"][0]
        self.assertEqual(seg["over_threshold_count"], 0)
        self.assertEqual(seg["inserted_count"], 1)
        self.assertEqual(
            (seg["beat_number_start"], seg["beat_number_end"]), (1, 2)
        )

        # 容差 1：插入音（0..1）与拖拍簇（3..6）边缘差 2 > 1 -> 分裂
        r = self.client.get(
            f"/jobs/{job_id}/review-segments?gap_tolerance_beats=1"
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["segment_count"], 2)
        self.assertEqual(data["segments"][0]["inserted_count"], 1)
        self.assertEqual(
            (data["segments"][0]["beat_number_start"],
             data["segments"][0]["beat_number_end"]),
            (1, 2),
        )
        self.assertEqual(data["segments"][1]["over_threshold_count"], 4)
        self.assertEqual(
            (data["segments"][1]["beat_number_start"],
             data["segments"][1]["beat_number_end"]),
            (4, 7),
        )

    def test_review_segments_clean_job_empty(self):
        job_id = self._create_done_job({
            "reference_beats": [0.5 * i for i in range(6)],
            "performance_onsets": [0.5 * i for i in range(6)],
        })
        r = self.client.get(f"/jobs/{job_id}/review-segments")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["segment_count"], 0)
        self.assertEqual(data["segments"], [])

    def test_review_segments_error_states(self):
        # 404：任务不存在
        r = self.client.get("/jobs/999999/review-segments")
        self.assertEqual(r.status_code, 404)

        # 409：任务尚未对齐（无对齐结果）
        r = self.client.post("/jobs", json={
            "reference_beats": REF, "performance_onsets": REF,
        })
        pending_id = r.json()["id"]
        r = self.client.get(f"/jobs/{pending_id}/review-segments")
        self.assertEqual(r.status_code, 409)
        self.assertIn("对齐", r.json()["detail"])

        # 409：状态为 done 但 result 缺失（遗留数据）-> 明确错误
        from app import crud, database
        conn = database.connect()
        try:
            crud.save_result(conn, pending_id, {})
            conn.execute(
                "UPDATE jobs SET status='done', result=NULL WHERE id=?",
                (pending_id,),
            )
            conn.commit()
        finally:
            conn.close()
        r = self.client.get(f"/jobs/{pending_id}/review-segments")
        self.assertEqual(r.status_code, 409)
        self.assertIn("尚无对齐结果", r.json()["detail"])

        # 400：上次对齐失败
        conn = database.connect()
        try:
            crud.save_error(conn, pending_id, "boom")
        finally:
            conn.close()
        r = self.client.get(f"/jobs/{pending_id}/review-segments")
        self.assertEqual(r.status_code, 400)
        self.assertIn("boom", r.json()["detail"])

    def test_review_segments_param_validation(self):
        job_id = self._create_done_job({
            "reference_beats": [0.5 * i for i in range(6)],
            "performance_onsets": [0.5 * i for i in range(6)],
        })
        r = self.client.get(
            f"/jobs/{job_id}/review-segments?deviation_threshold=0"
        )
        self.assertEqual(r.status_code, 422)
        r = self.client.get(
            f"/jobs/{job_id}/review-segments?deviation_threshold=-0.1"
        )
        self.assertEqual(r.status_code, 422)
        r = self.client.get(
            f"/jobs/{job_id}/review-segments?gap_tolerance_beats=-1"
        )
        self.assertEqual(r.status_code, 422)

    # ---- 跨任务合奏一致性 POST /ensembles/consensus ----

    CONSENSUS_REF = [0.0, 0.5, 1.0, 1.5, 2.0]

    def _create_ensemble_track(self, delay, missed_refs=(), beat_shift=None):
        """创建并对齐一条合奏轨；可指定固定延迟、漏拍与单拍演奏偏移。

        beat_shift=(ref_index, delta)：在匹配后手工把该拍 perf_time 偏移，
        再重新保存结果（模拟无法被 DTW 重对齐的单拍偏差）。
        """
        perf = [
            0.5 * i + delay
            for i in range(5) if i not in missed_refs
        ]
        body = {
            "reference_beats": self.CONSENSUS_REF,
            "performance_onsets": perf,
            "params": {"band_width": 3},
            "anchors": [
                {"ref_index": 0, "perf_index": 0},
                {"ref_index": 4, "perf_index": 4 - len(missed_refs)},
            ],
        }
        job_id = self._create_done_job(body)
        if beat_shift is not None:
            from app import crud, database
            ri, delta = beat_shift
            conn = database.connect()
            try:
                job = crud.get_job(conn, job_id)
                result = job["result"]
                for beat in result["matched_beats"]:
                    if beat["ref_index"] == ri:
                        beat["perf_time"] = round(beat["perf_time"] + delta, 6)
                        beat["deviation"] = round(
                            beat["perf_time"] - beat["ref_time"], 6
                        )
                        beat["status"] = (
                            "early" if delta < 0 else "late"
                        )
                crud.save_result(conn, job_id, result)
            finally:
                conn.close()
        return job_id

    def test_consensus_constant_delays(self):
        # 延迟须足够小（远小于漏拍惩罚），DTW 才会逐拍一一对应
        ids = [
            self._create_ensemble_track(0.01),
            self._create_ensemble_track(0.02),
            self._create_ensemble_track(0.03),
        ]
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": ids, "deviation_threshold": 0.05,
        })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["job_ids"], ids)
        self.assertEqual(data["track_count"], 3)
        self.assertEqual(data["reference_beats"], self.CONSENSUS_REF)
        self.assertAlmostEqual(data["tracks"][0]["delay"], 0.01, places=6)
        self.assertAlmostEqual(data["tracks"][1]["delay"], 0.02, places=6)
        self.assertAlmostEqual(data["tracks"][2]["delay"], 0.03, places=6)
        self.assertEqual(data["decidable_beat_count"], 5)
        self.assertEqual(data["undecidable_beat_count"], 0)
        self.assertEqual(data["outlier_count"], 0)
        for beat in data["beats"]:
            self.assertTrue(beat["decidable"])
            self.assertEqual(beat["matched_track_count"], 3)
            self.assertAlmostEqual(
                beat["consensus_time"], beat["ref_time"], places=6
            )
            self.assertEqual(beat["outliers"], [])
        self.assertEqual(data["parameters"]["min_tracks_per_beat"], 3)

    def test_consensus_outlier_and_threshold_boundary(self):
        ids = [
            self._create_ensemble_track(0.0),
            self._create_ensemble_track(0.0),
            # 第 4 拍（索引 3）拖拍 0.09，中位数延迟仍为 0
            self._create_ensemble_track(
                0.0, beat_shift=(3, 0.09)
            ),
        ]
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": ids, "deviation_threshold": 0.05,
        })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["outlier_count"], 1)
        beat = data["beats"][3]
        self.assertEqual(beat["outlier_count"], 1)
        self.assertEqual(beat["outliers"][0]["job_id"], ids[2])
        self.assertEqual(beat["outliers"][0]["status"], "late")
        self.assertAlmostEqual(beat["outliers"][0]["deviation"], 0.09, places=6)

        # 阈值放宽到 0.09：恰好等于偏差，严格大于才报 -> 无异常
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": ids, "deviation_threshold": 0.09,
        })
        self.assertEqual(r.json()["outlier_count"], 0)

    def test_consensus_undecidable_beat_lists_missing_per_track(self):
        # 末拍由锚点固定匹配，漏拍须取中间拍（索引 2）
        ids = [
            self._create_ensemble_track(0.0),
            self._create_ensemble_track(0.0, missed_refs=(2,)),
            self._create_ensemble_track(0.0, missed_refs=(2,)),
        ]
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": ids, "deviation_threshold": 0.05,
        })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["decidable_beat_count"], 4)
        self.assertEqual(data["undecidable_beat_count"], 1)
        bad = data["beats"][2]
        self.assertFalse(bad["decidable"])
        self.assertIsNone(bad["consensus_time"])
        self.assertEqual(bad["matched_track_count"], 1)
        self.assertEqual(bad["matched_job_ids"], [ids[0]])
        self.assertEqual(bad["missing_job_ids"], [ids[1], ids[2]])
        self.assertEqual(bad["outliers"], [])
        # 漏拍逐轨列出
        self.assertEqual(data["tracks"][0]["missed_beats"], [])
        self.assertEqual(
            [b["ref_index"] for b in data["tracks"][1]["missed_beats"]], [2]
        )
        self.assertEqual(
            [b["ref_index"] for b in data["tracks"][2]["missed_beats"]], [2]
        )
        self.assertEqual(
            data["tracks"][1]["missed_beats"][0]["beat_number"], 3
        )

    def test_consensus_validation_errors(self):
        ids = [
            self._create_ensemble_track(0.0),
            self._create_ensemble_track(0.0),
            self._create_ensemble_track(0.0),
        ]

        # 重复 ID -> 400
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": [ids[0], ids[0], ids[1]],
            "deviation_threshold": 0.05,
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("重复", r.json()["detail"])

        # 少于 3 个 ID -> 422（Pydantic min_length）
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": [ids[0], ids[1]],
            "deviation_threshold": 0.05,
        })
        self.assertEqual(r.status_code, 422)

        # 阈值非正 -> 422
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": ids, "deviation_threshold": 0,
        })
        self.assertEqual(r.status_code, 422)

        # 任务不存在 -> 404
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": [ids[0], ids[1], 999999],
            "deviation_threshold": 0.05,
        })
        self.assertEqual(r.status_code, 404)

        # 任务未对齐 -> 409
        pending = self.client.post("/jobs", json={
            "reference_beats": self.CONSENSUS_REF,
            "performance_onsets": self.CONSENSUS_REF,
        }).json()["id"]
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": [ids[0], ids[1], pending],
            "deviation_threshold": 0.05,
        })
        self.assertEqual(r.status_code, 409)

        # 上次对齐失败 -> 400
        from app import crud, database
        conn = database.connect()
        try:
            crud.save_error(conn, pending, "boom")
        finally:
            conn.close()
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": [ids[0], ids[1], pending],
            "deviation_threshold": 0.05,
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("boom", r.json()["detail"])

    def test_consensus_grid_mismatch_422(self):
        good = [
            self._create_ensemble_track(0.0),
            self._create_ensemble_track(0.0),
        ]
        # 不同的参考网格
        other = self._create_done_job({
            "reference_beats": [0.0, 0.6, 1.2, 1.8, 2.4],
            "performance_onsets": [0.0, 0.6, 1.2, 1.8, 2.4],
        })
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": [good[0], good[1], other],
            "deviation_threshold": 0.05,
        })
        self.assertEqual(r.status_code, 422)
        self.assertIn("参考节拍网格", r.json()["detail"])

        # 等长但数值不同也算不一致
        other2 = self._create_done_job({
            "reference_beats": [0.0, 0.5, 1.001, 1.5, 2.0],
            "performance_onsets": [0.0, 0.5, 1.001, 1.5, 2.0],
        })
        r = self.client.post("/ensembles/consensus", json={
            "job_ids": [good[0], good[1], other2],
            "deviation_threshold": 0.05,
        })
        self.assertEqual(r.status_code, 422)

    # ---- 多轨剪辑方案 POST /takes/comp-plan ----

    COMP_REF = [0.5 * i for i in range(6)]  # 6 拍

    def _create_comp_take(self, deviations):
        """创建并对齐一条剪辑轨：deviations 给出每个参考拍的偏差（秒）。

        全部拍一一匹配（perftime 严格递增要求偏差差不越过 0.5s 拍间隔）。
        """
        perf = [round(0.5 * i + d, 6) for i, d in enumerate(deviations)]
        return self._create_done_job({
            "reference_beats": self.COMP_REF,
            "performance_onsets": perf,
            "params": {"band_width": 2},
            "anchors": [
                {"ref_index": 0, "perf_index": 0},
                {"ref_index": 5, "perf_index": 5},
            ],
        })

    def _create_comp_take_with_miss(self, miss_index, deviations):
        """创建并对齐一条含漏拍的剪辑轨。"""
        perf = [
            round(0.5 * i + d, 6)
            for i, d in enumerate(deviations) if i != miss_index
        ]
        return self._create_done_job({
            "reference_beats": self.COMP_REF,
            "performance_onsets": perf,
            "params": {"band_width": 2},
            "anchors": [
                {"ref_index": 0, "perf_index": 0},
                {"ref_index": 5, "perf_index": 4},
            ],
        })

    def test_comp_plan_selects_and_merges(self):
        # 小节0（拍0-2）：轨1 更准（0.01）；小节1（拍3-5）：轨2 更准（0.01）
        id1 = self._create_comp_take([0.01] * 3 + [0.04] * 3)
        id2 = self._create_comp_take([0.03] * 3 + [0.01] * 3)
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, id2],
            "beats_per_bar": 3,
            "switch_cost": 0.0,
        })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["job_ids"], [id1, id2])
        self.assertEqual(data["bar_count"], 2)
        self.assertEqual(data["beats_per_bar"], 3)
        self.assertEqual(data["take_count"], 2)
        self.assertEqual(data["reference_beat_count"], 6)
        self.assertEqual(data["switch_count"], 1)
        self.assertAlmostEqual(data["total_cost"], 0.02, places=6)
        self.assertEqual(data["segment_count"], 2)
        first, second = data["segments"]
        self.assertEqual(first["job_id"], id1)
        self.assertEqual(second["job_id"], id2)
        self.assertEqual(first["bar_start"], 1)
        self.assertEqual(first["bar_end"], 1)
        self.assertEqual(second["bar_start"], 2)
        self.assertEqual(second["bar_end"], 2)
        # 参考拍索引范围 / 拍号
        self.assertEqual(first["ref_start_index"], 0)
        self.assertEqual(first["ref_end_index"], 2)
        self.assertEqual(first["beat_number_start"], 1)
        self.assertEqual(first["beat_number_end"], 3)
        # 来源首末匹配音时间
        self.assertAlmostEqual(first["source_start_time"], 0.01, places=6)
        self.assertAlmostEqual(first["source_end_time"], 1.01, places=6)
        self.assertAlmostEqual(second["source_start_time"], 1.51, places=6)
        self.assertAlmostEqual(second["source_end_time"], 2.51, places=6)
        self.assertEqual(first["source_start_perf_index"], 0)
        self.assertEqual(first["source_end_perf_index"], 2)
        self.assertEqual(second["source_start_perf_index"], 3)
        self.assertEqual(second["source_end_perf_index"], 5)

    def test_comp_plan_consecutive_same_source_merged(self):
        # 3 小节（每小节 2 拍）：轨1 占首节、轨2 占后两节且同源 -> 合并
        id1 = self._create_comp_take([0.01, 0.01, 0.09, 0.09, 0.03, 0.03])
        id2 = self._create_comp_take([0.08, 0.08, 0.02, 0.02, 0.02, 0.02])
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, id2],
            "beats_per_bar": 2,
            "switch_cost": 0.0,
        })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["bar_count"], 3)
        self.assertEqual([s["job_id"] for s in data["segments"]], [id1, id2])
        merged = data["segments"][1]
        self.assertEqual(merged["bar_start"], 2)
        self.assertEqual(merged["bar_end"], 3)
        self.assertEqual(merged["bar_count"], 2)
        self.assertEqual(merged["ref_start_index"], 2)
        self.assertEqual(merged["ref_end_index"], 5)
        self.assertEqual(merged["beat_number_start"], 3)
        self.assertEqual(merged["beat_number_end"], 6)
        self.assertAlmostEqual(
            merged["source_start_time"], 1.02, places=6
        )
        self.assertAlmostEqual(
            merged["source_end_time"], 2.52, places=6
        )

    def test_comp_plan_switch_cost_changes_plan(self):
        id1 = self._create_comp_take([0.01] * 3 + [0.04] * 3)
        id2 = self._create_comp_take([0.03] * 3 + [0.01] * 3)
        # 换轨 1->2 = 0.01+0.01+0.05 = 0.07 > 全程轨2 的 0.04 -> 全程轨2
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, id2],
            "beats_per_bar": 3,
            "switch_cost": 0.05,
        })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["segment_count"], 1)
        self.assertEqual(data["segments"][0]["job_id"], id2)
        self.assertEqual(data["switch_count"], 0)
        self.assertAlmostEqual(data["total_cost"], 0.04, places=6)

    def test_comp_plan_tie_prefers_request_order(self):
        id1 = self._create_comp_take([0.01] * 6)
        id2 = self._create_comp_take([0.01] * 6)
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, id2],
            "beats_per_bar": 3,
            "switch_cost": 0.02,
        })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual([s["job_id"] for s in data["segments"]], [id1])

        # 调换请求顺序 -> 优先另一条
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id2, id1],
            "beats_per_bar": 3,
            "switch_cost": 0.02,
        })
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(
            [s["job_id"] for s in r.json()["segments"]], [id2]
        )

    def test_comp_plan_uncovered_bar_422_with_bar_number(self):
        # 小节1（拍 3..5）两轨都漏第 4 拍（索引 3）
        id1 = self._create_comp_take_with_miss(
            3, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        id2 = self._create_comp_take_with_miss(
            3, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, id2],
            "beats_per_bar": 3,
            "switch_cost": 0.0,
        })
        self.assertEqual(r.status_code, 422)
        self.assertIn("2", r.json()["detail"])
        self.assertIn("小节", r.json()["detail"])

    def test_comp_plan_partial_match_bar_unavailable(self):
        # 轨1 漏小节0 的拍、轨2 漏小节1 的拍：每小节仍有完整轨 -> 各取一节
        id1 = self._create_comp_take_with_miss(
            3, [0.01] * 6
        )
        id2 = self._create_comp_take_with_miss(
            0, [0.02] * 6
        )
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, id2],
            "beats_per_bar": 3,
            "switch_cost": 0.0,
        })
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(
            [s["job_id"] for s in data["segments"]], [id1, id2]
        )
        self.assertEqual(data["switch_count"], 1)

    def test_comp_plan_validation_errors(self):
        id1 = self._create_comp_take([0.01] * 6)
        id2 = self._create_comp_take([0.02] * 6)

        # 重复 ID -> 400
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, id1], "beats_per_bar": 3, "switch_cost": 0.0,
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("重复", r.json()["detail"])

        # 少于 2 个 ID -> 422（Pydantic min_length）
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1], "beats_per_bar": 3, "switch_cost": 0.0,
        })
        self.assertEqual(r.status_code, 422)

        # beats_per_bar 非法 / switch_cost 为负 -> 422
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, id2], "beats_per_bar": 0, "switch_cost": 0.0,
        })
        self.assertEqual(r.status_code, 422)
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, id2], "beats_per_bar": 3, "switch_cost": -0.01,
        })
        self.assertEqual(r.status_code, 422)

        # 任务不存在 -> 404
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, 999999], "beats_per_bar": 3, "switch_cost": 0.0,
        })
        self.assertEqual(r.status_code, 404)

        # 任务未对齐 -> 409
        pending = self.client.post("/jobs", json={
            "reference_beats": self.COMP_REF,
            "performance_onsets": self.COMP_REF,
        }).json()["id"]
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, pending],
            "beats_per_bar": 3,
            "switch_cost": 0.0,
        })
        self.assertEqual(r.status_code, 409)

        # 上次对齐失败 -> 400
        from app import crud, database
        conn = database.connect()
        try:
            crud.save_error(conn, pending, "boom")
        finally:
            conn.close()
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, pending],
            "beats_per_bar": 3,
            "switch_cost": 0.0,
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("boom", r.json()["detail"])

    def test_comp_plan_grid_mismatch_422(self):
        id1 = self._create_comp_take([0.01] * 6)
        other = self._create_done_job({
            "reference_beats": [0.0, 0.6, 1.2, 1.8, 2.4, 3.0],
            "performance_onsets": [0.0, 0.6, 1.2, 1.8, 2.4, 3.0],
        })
        r = self.client.post("/takes/comp-plan", json={
            "job_ids": [id1, other],
            "beats_per_bar": 3,
            "switch_cost": 0.0,
        })
        self.assertEqual(r.status_code, 422)
        self.assertIn("参考节拍网格", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
