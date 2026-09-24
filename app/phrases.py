"""乐句出现检测（phrase occurrence）：在较长录音中按候选速度倍率搜索短乐句。

输入一段**短乐句的参考拍**（严格递增，秒）、一段**较长录音的起音时间**
（严格递增，秒）、若干**候选速度倍率**（正数，表示实际拍间隔与参考拍间隔
之比，如 2.0 表示录音以半速演奏该乐句）与非负**容差**（秒）：

1. **缩放与锚定**：按倍率 r 把参考拍换算为相对首拍的缩放间隔
   ``scaled[i] = (ref[i] − ref[0]) × r``；把参考首拍锚定在录音某起音 s 上，
   得到各参考拍的预测时间 ``pred[i] = onsets[s] + scaled[i]``——误差相对
   首次匹配起音计算，首拍误差恒为 0；
2. **候选生成**：对每个（倍率, 首起音 s, 尾起音 e）组合，在「参考拍索引与
   起音索引均严格递增、首末参考拍必须匹配、各匹配拍缩放后时间误差不超过
   容差」的约束下，用动态规划求出**匹配拍数最多、并列时总绝对误差最小**
   的映射；总匹配不足 3 拍的组合丢弃。参考拍允许漏拍（跳过中间参考拍），
   录音允许存在多余起音（不参与匹配）；
3. **全局选择**：候选时间区间 [onsets[s], onsets[e]] 互相重叠（含端点相接，
   即两次出现共用同一录音起音）时不可同时保留。按起音索引做区间调度 DP：
   先最大化互不重叠的出现次数，再最小化总绝对误差之和，最后按所选
   「(首起音, 倍率输入序号, 尾起音)」序列的字典序打破并列（首起音更早者
   优先；首起音相同则倍率在请求中出现顺序靠前者优先）。

结果按首起音时间升序返回每次出现的时间区间、倍率与参考拍 → 起音的索引
映射；没有任何命中时返回空列表。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any, Sequence

_DECIMALS = 6

# 容差比较的浮点余量：吸收边界上的二进制浮点噪声
_EPS = 1e-9

# 一次出现至少需要的匹配拍数（首拍、末拍及至少一拍中间拍）
MIN_MATCHED_BEATS = 3


class PhraseError(ValueError):
    """乐句出现检测输入非法。"""


def _round6(value: float) -> float:
    """四舍五入到 6 位小数并消除 -0.0（与结果经 JSON 存储的口径一致）。"""
    rounded = round(value, _DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _check_strictly_increasing(values: Sequence[float], label: str) -> None:
    if any(values[i] <= values[i - 1] for i in range(1, len(values))):
        raise PhraseError(f"{label}必须严格递增")


def _candidates_for_anchor(
    scaled: Sequence[float],
    onsets: Sequence[float],
    tolerance: float,
    rate_index: int,
    start: int,
) -> list[dict[str, Any]]:
    """枚举「参考首拍 ↔ onsets[start]」、指定倍率下的全部候选出现。

    对每个可作为末拍匹配的起音 e，用 DP 求该（首起音, 尾起音, 倍率）组合下
    匹配拍数最多、总绝对误差最小的映射；不足 MIN_MATCHED_BEATS 拍的丢弃。
    """
    k = len(scaled)
    anchor = onsets[start]
    preds = [anchor + s for s in scaled]
    limit = tolerance + _EPS

    # layers[i][j] = (matched_count, total_abs_error, parent)
    #   参考拍 i 匹配起音 j 时的最优前缀；parent = (参考拍 i', 起音 j')，
    #   首拍 parent 为 None。中间参考拍可跳过（前驱取自任意更早参考拍层）。
    layers: list[dict[int, tuple[int, float, tuple[int, int] | None]]] = [
        {start: (1, 0.0, None)}
    ]
    # 已处理各层的全部状态（按起音索引升序）：(起音 j, 拍数, 误差, 参考拍 i)
    prev_states: list[tuple[int, int, float, int]] = [(start, 1, 0.0, 0)]
    for i in range(1, k):
        lo = bisect_left(onsets, preds[i] - limit)
        hi = bisect_right(onsets, preds[i] + limit)
        layer: dict[int, tuple[int, float, tuple[int, int] | None]] = {}
        best: tuple[int, float, int, int] | None = None  # (拍数, 误差, j, i)
        p = 0
        for j in range(lo, hi):
            # 前驱起音索引必须严格小于 j（起音索引与参考拍索引均严格递增）
            while p < len(prev_states) and prev_states[p][0] < j:
                pj, pcount, pcost, pi = prev_states[p]
                if best is None or (pcount, -pcost) > (best[0], -best[1]):
                    best = (pcount, pcost, pj, pi)
                p += 1
            if best is None:
                continue
            err = abs(onsets[j] - preds[i])
            layer[j] = (
                best[0] + 1,
                _round6(best[1] + err),
                (best[3], best[2]),
            )
        layers.append(layer)
        # 把本层状态并入前驱池（两个升序序列归并）
        new_states = [(j, state[0], state[1], i) for j, state in layer.items()]
        merged: list[tuple[int, int, float, int]] = []
        a = b = 0
        while a < len(prev_states) and b < len(new_states):
            if prev_states[a][0] <= new_states[b][0]:
                merged.append(prev_states[a])
                a += 1
            else:
                merged.append(new_states[b])
                b += 1
        merged.extend(prev_states[a:])
        merged.extend(new_states[b:])
        prev_states = merged

    candidates = []
    for end, state in sorted(layers[k - 1].items()):
        count, cost, _ = state
        if count < MIN_MATCHED_BEATS:
            continue
        # 回溯参考拍 → 起音的映射（首拍误差恒为 0）
        pairs = []
        i, j = k - 1, end
        while True:
            pairs.append((i, j, onsets[j] - preds[i]))
            parent = layers[i][j][2]
            if parent is None:
                break
            i, j = parent
        pairs.reverse()
        candidates.append({
            "rate_index": rate_index,
            "start": start,
            "end": end,
            "matched_count": count,
            "total_abs_error": cost,
            "pairs": pairs,
        })
    return candidates


def find_occurrences(
    reference_beats: Sequence[float],
    recording_onsets: Sequence[float],
    rates: Sequence[float],
    tolerance: float,
) -> dict[str, Any]:
    """在录音起音中按候选倍率搜索乐句出现，返回全局选择后的出现列表。

    ``reference_beats`` 为短乐句参考拍（≥2 拍，严格递增，秒）；
    ``recording_onsets`` 为较长录音的起音时间（严格递增，秒）；``rates``
    为候选速度倍率（正数，实际拍间隔与参考拍间隔之比）；``tolerance`` 为
    非负容差（秒）。无命中时 ``occurrences`` 为空列表。
    """
    if len(reference_beats) < 2:
        raise PhraseError("参考拍至少需要 2 个（首拍与末拍）")
    _check_strictly_increasing(reference_beats, "参考拍")
    if not recording_onsets:
        raise PhraseError("录音起音至少需要 1 个")
    _check_strictly_increasing(recording_onsets, "录音起音")
    if not rates:
        raise PhraseError("候选速度倍率至少需要 1 个")
    if any(r <= 0 for r in rates):
        raise PhraseError("候选速度倍率必须为正数")
    if tolerance < 0:
        raise PhraseError("容差 tolerance 不可为负")

    k = len(reference_beats)
    m = len(recording_onsets)
    refs = [float(t) for t in reference_beats]
    onsets = [float(t) for t in recording_onsets]

    # ---- 1+2. 逐倍率、逐首起音生成候选（同首同尾同倍率只留最优映射）----
    candidates: list[dict[str, Any]] = []
    for rate_index, rate in enumerate(rates):
        scaled = [(refs[i] - refs[0]) * rate for i in range(k)]
        for start in range(m):
            candidates.extend(
                _candidates_for_anchor(scaled, onsets, tolerance, rate_index, start)
            )

    # ---- 3. 全局选择：先最大化互不重叠出现数，再最小化总绝对误差，
    #         最后按 (首起音, 倍率输入序号, 尾起音) 序列字典序打破并列 ----
    by_start: list[list[int]] = [[] for _ in range(m)]
    for ci, cand in enumerate(candidates):
        by_start[cand["start"]].append(ci)

    # best_key[i]：只用首起音 ≥ i 的候选时的最优解
    #   key = (−出现次数, 总绝对误差, (首起音, 倍率序号, 尾起音) 序列)
    # best_pick[i]：达成最优解的候选索引序列（区间端点相接即视为重叠，
    #   因此选中候选的下一候选首起音索引必须严格大于其尾起音索引）
    best_key: list[tuple[int, float, tuple] | None] = [None] * (m + 1)
    best_pick: list[tuple[int, ...] | None] = [None] * (m + 1)
    best_key[m] = (0, 0.0, ())
    best_pick[m] = ()
    for i in range(m - 1, -1, -1):
        key = best_key[i + 1]
        pick = best_pick[i + 1]
        for ci in by_start[i]:
            cand = candidates[ci]
            tail_key = best_key[cand["end"] + 1]
            tail_pick = best_pick[cand["end"] + 1]
            ck = (
                tail_key[0] - 1,
                _round6(tail_key[1] + cand["total_abs_error"]),
                ((cand["start"], cand["rate_index"], cand["end"]),) + tail_key[2],
            )
            if key is None or ck < key:
                key = ck
                pick = (ci,) + tail_pick
        best_key[i] = key
        best_pick[i] = pick

    # ---- 输出（按首起音时间升序）----
    occurrences = []
    for order, ci in enumerate(best_pick[0] or ()):
        cand = candidates[ci]
        occurrences.append({
            "occurrence_index": order,
            "rate": float(rates[cand["rate_index"]]),
            "rate_index": cand["rate_index"],
            "start_time": _round6(onsets[cand["start"]]),
            "end_time": _round6(onsets[cand["end"]]),
            "start_onset_index": cand["start"],
            "end_onset_index": cand["end"],
            "matched_count": cand["matched_count"],
            "total_absolute_error": _round6(cand["total_abs_error"]),
            "mapping": [
                {
                    "ref_index": ri,
                    "onset_index": oi,
                    "error": _round6(err),
                }
                for ri, oi, err in cand["pairs"]
            ],
        })

    return {
        "parameters": {
            "tolerance": tolerance,
            "rates": [float(r) for r in rates],
            "reference_beat_count": k,
            "recording_onset_count": m,
            "min_matched_beats": MIN_MATCHED_BEATS,
        },
        "candidate_count": len(candidates),
        "occurrence_count": len(occurrences),
        "occurrences": occurrences,
    }
