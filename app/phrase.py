"""乐句出现搜索（phrase occurrence search）：在较长录音中按候选倍率定位短乐句。

输入短乐句的**参考拍**（严格递增，秒）、较长录音的**起音时间**（严格递增，
秒）、候选**速度倍率**（正数，表示实际拍间隔与参考拍间隔之比，如 2.0 表示
慢一倍）与非负**容差**（秒）。处理流程：

1. **候选生成**：对每个倍率 ``k``（按输入顺序）与每个起音 ``j0``（作为参考
   首拍的匹配），用二分查找圈出落在
   ``ons[j0] + k*(ref[-1] − ref[0]) ± 容差`` 内的起音 ``j1`` 作为参考尾拍
   的匹配。误差一律**相对首次匹配起音**计算：参考第 i 拍的期望时间为
   ``ons[j0] + k*(ref[i] − ref[0])``，误差 = 实际起音 − 期望时间。
2. **候选内最优映射**：在固定首尾配对 ``(0↔j0, n−1↔j1)`` 下，中间参考拍
   ``1..n−2`` 与起音 ``j0+1..j1−1`` 之间做单调子序列匹配 DP——允许漏拍
   （参考拍不匹配）与多余起音（起音不使用），每个匹配拍的 |误差| 不得超过
   容差。同一首尾和倍率下先**最大化匹配拍数**、再**最小化总绝对误差**；
   首尾参考拍必须匹配且总匹配拍数不得少于 3，否则候选不成立。
3. **全局选择**：候选的时间区间为 ``[ons[j0], ons[j1]]``，区间重叠
   （严格相交；端点相接不算重叠，允许前一出现的尾音同时是后一出现的首音）
   的候选不可兼得。按字典序目标做加权区间调度 DP：先**最大化互不重叠的
   出现次数**，再**最小化总绝对误差**（各出现误差之和），再按**所选起点
   序列**（字典序，起点越早越优先）与**倍率输入顺序**（下标小者优先）
   打破并列。

无任何命中时返回空的出现列表（仍为正常结果）。本模块为纯计算，不读写
数据库、不修改任何已有任务。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any, Sequence

_DECIMALS = 6

# 一次出现至少需要的匹配拍数（首尾参考拍必须匹配，故乐句不足 3 拍时不可能命中）
MIN_MATCHED_BEATS = 3

# 二分查找窗口的浮点余量（大于 6 位小数舍入漂移 5e-7，保证舍入后可行的
# 尾拍一定落在窗口内，最终可行性仍以舍入后的误差判定为准）
_WINDOW_EPS = 1e-6


class PhraseError(ValueError):
    """乐句出现搜索输入非法。"""


def _round6(value: float) -> float:
    """四舍五入到 6 位小数并消除 -0.0（与结果 JSON 口径一致）。"""
    rounded = round(value, _DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _validate_inputs(
    reference_beats: Sequence[float],
    performance_onsets: Sequence[float],
    ratios: Sequence[float],
    tolerance: float,
) -> None:
    if not reference_beats:
        raise PhraseError("参考拍 reference_beats 不可为空")
    if not performance_onsets:
        raise PhraseError("起音序列 performance_onsets 不可为空")
    if not ratios:
        raise PhraseError("候选倍率 ratios 不可为空")
    for name, values in (
        ("参考拍 reference_beats", reference_beats),
        ("起音序列 performance_onsets", performance_onsets),
    ):
        if any(values[i] <= values[i - 1] for i in range(1, len(values))):
            raise PhraseError(f"{name}必须严格递增")
    if any(k <= 0 for k in ratios):
        raise PhraseError("候选倍率 ratios 必须全部为正数")
    if tolerance < 0:
        raise PhraseError("容差 tolerance 不可为负")


def _best_inner_mapping(
    ref: Sequence[float],
    ons: Sequence[float],
    ratio: float,
    j0: int,
    j1: int,
    tolerance: float,
) -> tuple[list[tuple[int, int, float]], float] | None:
    """固定首尾配对 (0↔j0, n−1↔j1)，为中间参考拍求最优子序列匹配。

    目标按字典序：先最大化匹配拍数，再最小化总绝对误差（各匹配拍 6 位小数
    舍入误差绝对值之和）。返回 ``(mapping, total_abs_error)``，其中 mapping
    为 ``(ref_index, onset_index, error)`` 列表（含首尾）；总匹配拍数不足
    ``MIN_MATCHED_BEATS`` 时返回 ``None``。
    """
    n = len(ref)
    base = ons[j0]  # 首次匹配起音：所有误差的零点
    origin = ref[0]

    def error_at(i: int, j: int) -> float:
        expected = base + ratio * (ref[i] - origin)
        return _round6(ons[j] - expected)

    last_error = error_at(n - 1, j1)
    if abs(last_error) > tolerance:
        return None

    # 中间参考拍 1..n−2（局部 0..B−1）× 中间起音 j0+1..j1−1（局部 0..W−1）
    n_inner = n - 2
    width = j1 - j0 - 1

    # dp[b][w] = (匹配数, 总绝对误差)：参考拍 1..b、起音 j0+1..j0+w 内的最优
    dp = [[(0, 0.0)] * (width + 1) for _ in range(n_inner + 1)]
    # choice[b][w]：'match' / 'skip_onset' / 'skip_beat'（同优时按此序优先）
    choice = [[""] * (width + 1) for _ in range(n_inner + 1)]

    for b in range(1, n_inner + 1):
        i = b  # 参考拍索引
        for w in range(1, width + 1):
            j = j0 + w  # 起音索引
            # 候选转移按优先级排列：匹配 > 跳过该起音 > 跳过该参考拍（漏拍）
            options: list[tuple[int, float, int, str]] = []
            err = error_at(i, j)
            if abs(err) <= tolerance:
                prev_count, prev_cost = dp[b - 1][w - 1]
                options.append(
                    # 误差均为 6 位小数，累加后再舍入可消除浮点尾数，
                    # 保证同值比较是精确的十进制比较
                    (prev_count + 1, _round6(prev_cost + abs(err)), 0, "match")
                )
            skip_onset = dp[b][w - 1]
            options.append((skip_onset[0], skip_onset[1], 1, "skip_onset"))
            skip_beat = dp[b - 1][w]
            options.append((skip_beat[0], skip_beat[1], 2, "skip_beat"))
            best = min(options, key=lambda o: (-o[0], o[1], o[2]))
            dp[b][w] = (best[0], best[1])
            choice[b][w] = best[3]

    inner_count, inner_cost = dp[n_inner][width]
    matched_count = inner_count + 2  # 加上首尾两个必匹配拍
    if matched_count < MIN_MATCHED_BEATS:
        return None

    # 回溯中间拍的匹配对
    pairs: list[tuple[int, int, float]] = []
    b, w = n_inner, width
    while b > 0 and w > 0:
        step = choice[b][w]
        if step == "match":
            i, j = b, j0 + w
            pairs.append((i, j, error_at(i, j)))
            b, w = b - 1, w - 1
        elif step == "skip_onset":
            w -= 1
        else:  # skip_beat：该参考拍为漏拍
            b -= 1
    pairs.reverse()

    mapping = [(0, j0, 0.0), *pairs, (n - 1, j1, last_error)]
    total_abs_error = _round6(inner_cost + abs(last_error))
    return mapping, total_abs_error


def _selection_key(selection: list[dict[str, Any]]) -> tuple:
    """全局选择的字典序目标（越小越优）。

    依次：出现次数多者优先、总绝对误差小者优先、起点序列字典序小者优先、
    倍率输入顺序靠前者优先、结束时间序列字典序小者优先（保证全序确定）。
    """
    return (
        -len(selection),
        _round6(sum(c["total_abs_error"] for c in selection)),
        tuple(c["start_time"] for c in selection),
        tuple(c["ratio_index"] for c in selection),
        tuple(c["end_time"] for c in selection),
    )


def _select_non_overlapping(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """对区间重叠的候选做全局选择（加权区间调度 DP）。

    候选按结束时间排序；``prev[i]`` 为与第 i 个候选相容（结束时间 ≤ 其开始
    时间，端点相接视为不重叠）的候选数量前缀。``best[i]`` 为只考虑前 i 个
    候选时的最优选择，按 ``_selection_key`` 比较。
    """
    ordered = sorted(
        candidates,
        key=lambda c: (c["end_time"], c["start_time"], c["ratio_index"]),
    )
    ends = [c["end_time"] for c in ordered]

    best: list[list[dict[str, Any]]] = [[] for _ in range(len(ordered) + 1)]
    for i in range(1, len(ordered) + 1):
        cand = ordered[i - 1]
        prev = bisect_right(ends, cand["start_time"], 0, i - 1)
        take = best[prev] + [cand]
        skip = best[i - 1]
        best[i] = min((skip, take), key=_selection_key)
    return best[len(ordered)]


def find_phrase_occurrences(
    reference_beats: Sequence[float],
    performance_onsets: Sequence[float],
    ratios: Sequence[float],
    tolerance: float,
) -> dict[str, Any]:
    """在录音起音中按候选倍率搜索乐句出现，返回全局选择后的出现列表。

    ``reference_beats`` / ``performance_onsets`` 为严格递增的秒级时间；
    ``ratios`` 为正的候选倍率（实际拍间隔 / 参考拍间隔），按输入顺序参与
    并列打破；``tolerance`` 为非负容差（秒），各匹配拍误差相对首次匹配
    起音计算。
    """
    _validate_inputs(reference_beats, performance_onsets, ratios, tolerance)

    ref = [float(t) for t in reference_beats]
    ons = [float(t) for t in performance_onsets]
    n, m = len(ref), len(ons)

    occurrences: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    # 参考拍或起音不足 3 个时不可能满足“总匹配不少于三拍”，直接无候选
    if n >= MIN_MATCHED_BEATS and m >= MIN_MATCHED_BEATS:
        ref_span = ref[-1] - ref[0]
        for ratio_index, ratio in enumerate(ratios):
            k = float(ratio)
            span = k * ref_span
            for j0 in range(m - 1):
                expected_end = ons[j0] + span
                lo = bisect_left(ons, expected_end - tolerance - _WINDOW_EPS, j0 + 1)
                hi = bisect_right(ons, expected_end + tolerance + _WINDOW_EPS, j0 + 1)
                for j1 in range(lo, hi):
                    result = _best_inner_mapping(ref, ons, k, j0, j1, tolerance)
                    if result is None:
                        continue
                    mapping, total_abs_error = result
                    candidates.append({
                        "ratio": k,
                        "ratio_index": ratio_index,
                        "start_onset_index": j0,
                        "end_onset_index": j1,
                        "start_time": ons[j0],
                        "end_time": ons[j1],
                        "matched_count": len(mapping),
                        "total_abs_error": total_abs_error,
                        "mapping": mapping,
                    })

        occurrences = _select_non_overlapping(candidates)

    out_occurrences: list[dict[str, Any]] = []
    for occurrence_index, cand in enumerate(occurrences):
        out_occurrences.append({
            "occurrence_index": occurrence_index,
            "ratio": cand["ratio"],
            "ratio_index": cand["ratio_index"],
            "start_time": _round6(cand["start_time"]),
            "end_time": _round6(cand["end_time"]),
            "start_onset_index": cand["start_onset_index"],
            "end_onset_index": cand["end_onset_index"],
            "matched_count": cand["matched_count"],
            "total_abs_error": cand["total_abs_error"],
            "mapping": [
                {
                    "ref_index": i,
                    "onset_index": j,
                    "ref_time": _round6(ref[i]),
                    "onset_time": _round6(ons[j]),
                    "error": err,
                }
                for i, j, err in cand["mapping"]
            ],
        })

    return {
        "parameters": {
            "tolerance": tolerance,
            "ratios": [float(k) for k in ratios],
            "min_matched_beats": MIN_MATCHED_BEATS,
        },
        "reference_beat_count": n,
        "onset_count": m,
        "candidate_count": len(candidates),
        "occurrence_count": len(out_occurrences),
        "occurrences": out_occurrences,
    }
