"""剪辑方案（compilation plan）：按小节从多条已对齐轨中挑录音做 DP 拼轨。

输入若干共享**完全相同**参考节拍网格的已完成任务（每个任务对应一条演奏
录音），以及**每小节拍数**（beats per bar）与非负的**换轨代价**：

1. 按参考拍把网格划分为连续小节，每小节占 ``beats_per_bar`` 拍（末尾不足
   一小节的尾部拍仍作为一个（不完整）小节处理）；
2. 某条录音只有**完整匹配该小节全部拍**时才可用于该小节（小节内任一拍为
   漏拍即不可用）；选用代价为该小节匹配拍**平均绝对偏差**
   （mean absolute deviation，单位秒）；
3. 用动态规划最小化「各小节选用代价之和 + 相邻小节换轨代价之和」。在所有
   总代价最小的方案中，取按任务序号**字典序最小**的逐小节选轨序列：前向
   DP 与出口后缀各自求出最小代价，再从首小节起贪心，每节取请求顺序最靠前、
   且「已选前缀 + 本小节 + 最优后缀」仍等于全局最优总代价的任务；
4. 任一存在的小节没有任何可用录音时报错（由接口转成含小节序号的 422）。

DP 解出每小节选用的任务后，把**相邻且同源**（同一任务）的小节合并为连续
片段返回，片段含任务 ID、参考拍索引范围（及 1 基拍号 / 小节序号范围）与
来源首末匹配音的演奏时间。
"""

from __future__ import annotations

from typing import Any, Sequence

_DECIMALS = 6

# 一次剪辑方案请求至少需要多少个任务
MIN_TAKES = 2


class CompPlanError(ValueError):
    """剪辑方案输入非法（含无可用录音的小节）。"""


def _round6(value: float) -> float:
    """四舍五入到 6 位小数并消除 -0.0（结果经 JSON 存储也是 6 位小数口径）。"""
    rounded = round(value, _DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def build_comp_plan(
    jobs: Sequence[dict[str, Any]],
    beats_per_bar: int,
    switch_cost: float,
) -> dict[str, Any]:
    """对已完成且参考网格完全相同的任务序列计算按小节拼轨的剪辑方案。

    ``jobs`` 为按请求顺序排列的 crud 任务字典（至少 2 个，``status='done'``
    且 ``reference_beats`` 完全一致）；``beats_per_bar`` 为每小节拍数（≥1），
    ``switch_cost`` 为相邻小节换轨代价（≥0，单位与秒级选用代价相权衡）。
    """
    if beats_per_bar < 1:
        raise CompPlanError("每小节拍数 beats_per_bar 必须为正整数")
    if switch_cost < 0:
        raise CompPlanError("换轨代价 switch_cost 不可为负")
    if len(jobs) < MIN_TAKES:
        raise CompPlanError(f"剪辑方案至少需要 {MIN_TAKES} 个已完成任务")

    reference_beats = jobs[0]["reference_beats"]
    n_ref = len(reference_beats)
    n_takes = len(jobs)

    # ---- 1. 按参考拍划小节 ----
    n_bars = (n_ref + beats_per_bar - 1) // beats_per_bar

    # bar_beats[b]：第 b 小节包含的参考拍索引（末尾小节可能不足一小节）
    bar_beats: list[list[int]] = [[] for _ in range(n_bars)]
    for ri in range(n_ref):
        bar_beats[ri // beats_per_bar].append(ri)

    # ---- 2. 逐轨逐小节计算可用性与选用代价（匹配拍平均绝对偏差）----
    # cost[(take, bar)] = 该小节的选用代价（秒）；缺键表示该录音不可用
    cost: dict[tuple[int, int], float] = {}
    for take_index, job in enumerate(jobs):
        matched = (job.get("result") or {}).get("matched_beats") or []
        # matched_by_ref[ri] = 该参考拍上的匹配拍（异常多配对时取最后一条）
        matched_by_ref: dict[int, dict[str, Any]] = {}
        for beat in matched:
            matched_by_ref[beat["ref_index"]] = beat

        for bar in range(n_bars):
            beats = bar_beats[bar]
            bar_matches = [matched_by_ref.get(ri) for ri in beats]
            if any(b is None for b in bar_matches):
                # 小节内存在漏拍 -> 该录音不可用于本小节
                continue
            total_abs = sum(abs(float(b["deviation"])) for b in bar_matches)
            cost[(take_index, bar)] = total_abs / len(beats)

    # 无可用录音的小节：422 明细中给出 1 基小节序号
    unavailable = [
        bar + 1
        for bar in range(n_bars)
        if not any((take, bar) in cost for take in range(n_takes))
    ]
    if unavailable:
        raise CompPlanError(
            "以下小节没有任何录音完整匹配其全部拍，无法生成剪辑方案："
            f"第 {'、'.join(str(b) for b in unavailable)} 小节"
        )

    # ---- 3. 动态规划选轨 ----
    # (a) 前向 DP：dp[bar][take] = 从首小节到第 bar 小节、且第 bar 小节选用
    #     take 的最小代价（不含从 take 换出到下一小节的换轨代价）。不可用为 inf。
    dp: list[list[float]] = [[float("inf")] * n_takes for _ in range(n_bars)]
    for take in range(n_takes):
        if (take, 0) in cost:
            dp[0][take] = cost[(take, 0)]
    for bar in range(1, n_bars):
        for take in range(n_takes):
            if (take, bar) not in cost:
                continue
            dp[bar][take] = min(
                dp[bar - 1][prev] + (0.0 if prev == take else switch_cost)
                for prev in range(n_takes)
            ) + cost[(take, bar)]
    total_cost = min(dp[n_bars - 1])

    # (b) 出口后缀（标量）：suffix[bar][take] = 第 bar 小节已定 take 时，换入
    #     第 bar+1 小节的换轨代价与第 bar+1..末小节代价之和的最小值；末小节为 0。
    suffix: list[list[float]] = [[0.0] * n_takes for _ in range(n_bars)]
    for bar in range(n_bars - 2, -1, -1):
        for take in range(n_takes):
            best_rest = float("inf")
            for nxt in range(n_takes):
                if (nxt, bar + 1) not in cost:
                    continue
                rest = (
                    (0.0 if take == nxt else switch_cost)
                    + cost[(nxt, bar + 1)]
                    + suffix[bar + 1][nxt]
                )
                if rest < best_rest:
                    best_rest = rest
            suffix[bar][take] = best_rest

    # (c) 从左到右贪心重建：每一小节取请求顺序最靠前、且“已选前缀代价 +
    #     换入代价 + 本小节选用代价 + 该状态最优出口后缀”恰好等于全局最优
    #     总代价的任务。后缀最优是标量可递归的，因此该可行性判定逐节成立；
    #     每节取最靠前的可行任务即得字典序最小的最优选轨序列。偏差输入为
    #     6 位小数，用容差判定代价相等。
    chosen: list[int] = []
    prefix_cost = 0.0
    prev_take: int | None = None
    for bar in range(n_bars):
        for take in range(n_takes):
            if (take, bar) not in cost:
                continue
            incoming = (
                0.0 if prev_take is None or prev_take == take else switch_cost
            )
            candidate = (
                prefix_cost + incoming + cost[(take, bar)] + suffix[bar][take]
            )
            if abs(candidate - total_cost) <= 1e-9:
                chosen.append(take)
                prefix_cost += incoming + cost[(take, bar)]
                prev_take = take
                break

    switch_count = sum(
        1 for bar in range(1, n_bars) if chosen[bar] != chosen[bar - 1]
    )

    # ---- 4. 合并连续同源小节为片段 ----
    segments: list[dict[str, Any]] = []
    group_start = 0
    for bar in range(1, n_bars + 1):
        if bar < n_bars and chosen[bar] == chosen[group_start]:
            continue
        take = chosen[group_start]
        start_ref = bar_beats[group_start][0]
        end_ref = bar_beats[bar - 1][-1]
        result = jobs[take].get("result") or {}
        matched_by_ref = {
            b["ref_index"]: b for b in (result.get("matched_beats") or [])
        }
        first_beat = matched_by_ref[start_ref]
        last_beat = matched_by_ref[end_ref]
        group_mads = [
            cost[(take, b)] for b in range(group_start, bar)
        ]
        group_bar_count = bar - group_start
        segments.append({
            "segment_index": len(segments),
            "job_id": jobs[take]["id"],
            "name": jobs[take].get("name"),
            "bar_start": group_start + 1,
            "bar_end": bar,
            "bar_count": group_bar_count,
            "ref_start_index": start_ref,
            "ref_end_index": end_ref,
            "beat_number_start": start_ref + 1,
            "beat_number_end": end_ref + 1,
            "beat_count": end_ref - start_ref + 1,
            "ref_start_time": _round6(float(reference_beats[start_ref])),
            "ref_end_time": _round6(float(reference_beats[end_ref])),
            "source_start_time": _round6(float(first_beat["perf_time"])),
            "source_end_time": _round6(float(last_beat["perf_time"])),
            "source_start_perf_index": first_beat["perf_index"],
            "source_end_perf_index": last_beat["perf_index"],
            "mean_absolute_deviation": _round6(
                sum(group_mads) / group_bar_count
            ),
            "selection_cost": _round6(sum(group_mads)),
        })
        group_start = bar

    return {
        "parameters": {
            "beats_per_bar": beats_per_bar,
            "switch_cost": switch_cost,
        },
        "job_ids": [job["id"] for job in jobs],
        "reference_beats": [_round6(float(t)) for t in reference_beats],
        "reference_beat_count": n_ref,
        "beats_per_bar": beats_per_bar,
        "bar_count": n_bars,
        "take_count": n_takes,
        "switch_count": switch_count,
        "total_cost": _round6(total_cost),
        "segment_count": len(segments),
        "segments": segments,
    }
