"""跨任务合奏一致性（ensemble consensus）分析。

输入若干已完成（对齐成功）且共享**完全相同**参考节拍网格的任务（每个任务
对应一条演奏轨），处理流程：

1. **固定录音延迟估计**：对每条轨，以其匹配拍 ``perf_time − ref_time`` 的
   中位数作为该轨的固定录音延迟；校正后的演奏时间为
   ``corrected = perf_time − delay``（中位数不受个别抢 / 拖拍影响）。
2. **逐拍合奏基准**：按参考拍汇总各轨校正时间，仅当**至少 3 轨**匹配该拍
   时取校正时间的中位数为合奏基准；校正时间相对基准的偏差**严格大于**
   异常阈值（秒）的轨列为异常（early / late）并给出任务 ID 与偏差。
3. **覆盖不足**：匹配轨数不足 3 的拍标记为“不可判定”，不给基准、不报异常；
   各轨漏拍在轨结果中逐轨列出（每拍另给缺失该拍的任务 ID 列表）。
"""

from __future__ import annotations

from typing import Any, Sequence

_DECIMALS = 6

# 每拍至少需要多少条匹配轨才能给出合奏基准
MIN_TRACKS_PER_BEAT = 3
# 一次合奏一致性请求至少需要多少个任务
MIN_JOBS = 3


class EnsembleError(ValueError):
    """合奏一致性输入非法。"""


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _round6(value: float) -> float:
    """四舍五入到 6 位小数并消除 -0.0（结果经 JSON 存储也是 6 位小数口径）。"""
    rounded = round(value, _DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def build_consensus(
    jobs: Sequence[dict[str, Any]],
    deviation_threshold: float,
) -> dict[str, Any]:
    """对已完成且参考网格完全相同的任务序列计算合奏一致性。

    ``jobs`` 为按提交顺序排列的 crud 任务字典（至少 3 个，``status='done'``
    且 ``reference_beats`` 完全一致）；``deviation_threshold`` 为秒级异常
    阈值（正数）。返回各轨延迟与逐拍合奏结果。
    """
    if deviation_threshold <= 0:
        raise EnsembleError("异常阈值 deviation_threshold 必须为正数")
    if len(jobs) < MIN_JOBS:
        raise EnsembleError(f"合奏一致性至少需要 {MIN_JOBS} 个已完成任务")

    reference_beats = jobs[0]["reference_beats"]
    n_ref = len(reference_beats)
    n_jobs = len(jobs)

    tracks: list[dict[str, Any]] = []
    # per_beat[ri] 保存 (轨序号, 校正时间)；按任务提交顺序追加，天然有序
    per_beat: list[list[tuple[int, float]]] = [[] for _ in range(n_ref)]

    # ---- 1. 逐轨估计固定录音延迟并校正演奏时间 ----
    for track_index, job in enumerate(jobs):
        result = job.get("result") or {}
        matched = result.get("matched_beats") or []

        offsets = [
            float(b["perf_time"]) - float(b["ref_time"]) for b in matched
        ]
        if offsets:
            delay = _median(offsets)
            delay_out: float | None = _round6(delay)
        else:
            # 没有匹配拍的轨无法估计延迟，也不参与任何拍的合奏基准
            delay = None
            delay_out = None

        for beat in matched:
            ri = beat["ref_index"]
            corrected = (
                float(beat["perf_time"]) - delay
                if delay is not None
                else float(beat["perf_time"])
            )
            per_beat[ri].append((track_index, corrected))

        missed = sorted(
            result.get("missed_beats") or [], key=lambda b: b["ref_index"]
        )
        tracks.append({
            "job_id": job["id"],
            "name": job.get("name"),
            "delay": delay_out,
            "matched_beat_count": len(matched),
            "missed_beat_count": len(missed),
            "missed_beats": [
                {
                    "ref_index": beat["ref_index"],
                    "beat_number": beat["ref_index"] + 1,
                    # 以共享参考网格的时间为准（所有轨网格完全一致）
                    "ref_time": _round6(float(reference_beats[beat["ref_index"]])),
                }
                for beat in missed
            ],
        })

    # ---- 2/3. 逐拍汇总：≥3 轨取中位数基准并报异常，否则不可判定 ----
    beats: list[dict[str, Any]] = []
    decidable_count = 0
    total_outliers = 0

    for ri, ref_time in enumerate(reference_beats):
        entries = per_beat[ri]
        matched_track_indices = [t for t, _ in entries]
        matched_set = set(matched_track_indices)
        matched_job_ids = [jobs[t]["id"] for t in matched_track_indices]
        missing_job_ids = [
            jobs[t]["id"] for t in range(n_jobs) if t not in matched_set
        ]

        if len(entries) >= MIN_TRACKS_PER_BEAT:
            baseline = _median([corrected for _, corrected in entries])
            outliers: list[dict[str, Any]] = []
            for track_index, corrected in entries:
                # 与结果的 6 位小数 JSON 存储口径一致，避免浮点尾数
                # （如 1.59−1.5=0.08999999999999986）影响阈值边界判定
                deviation = _round6(corrected - baseline)
                if abs(deviation) > deviation_threshold:
                    outliers.append({
                        "job_id": jobs[track_index]["id"],
                        "corrected_time": _round6(corrected),
                        "deviation": deviation,
                        "status": "early" if deviation < 0 else "late",
                    })
            beats.append({
                "ref_index": ri,
                "beat_number": ri + 1,
                "ref_time": _round6(float(ref_time)),
                "decidable": True,
                "matched_track_count": len(entries),
                "consensus_time": _round6(baseline),
                "matched_job_ids": matched_job_ids,
                "missing_job_ids": missing_job_ids,
                "outlier_count": len(outliers),
                "outliers": outliers,
            })
            decidable_count += 1
            total_outliers += len(outliers)
        else:
            # 覆盖不足：不可判定，不给基准、不报异常
            beats.append({
                "ref_index": ri,
                "beat_number": ri + 1,
                "ref_time": _round6(float(ref_time)),
                "decidable": False,
                "matched_track_count": len(entries),
                "consensus_time": None,
                "matched_job_ids": matched_job_ids,
                "missing_job_ids": missing_job_ids,
                "outlier_count": 0,
                "outliers": [],
            })

    return {
        "parameters": {
            "deviation_threshold": deviation_threshold,
            "min_tracks_per_beat": MIN_TRACKS_PER_BEAT,
        },
        "job_ids": [job["id"] for job in jobs],
        "reference_beats": [_round6(float(t)) for t in reference_beats],
        "reference_beat_count": n_ref,
        "track_count": n_jobs,
        "tracks": tracks,
        "beat_count": n_ref,
        "decidable_beat_count": decidable_count,
        "undecidable_beat_count": n_ref - decidable_count,
        "outlier_count": total_outliers,
        "beats": beats,
    }
