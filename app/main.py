"""多轨录音节拍对齐 API。

启动：uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from . import crud
from .alignment import (
    AlignmentError,
    AlignmentParams,
    align_beats,
    validate_anchors,
)
from .comp_plan import CompPlanError, build_comp_plan
from .database import get_db, init_db
from .ensemble import EnsembleError, build_consensus
from .review import ReviewError, extract_review_segments
from .schemas import (
    CompPlanOut,
    CompPlanRequest,
    ConsensusRequest,
    EnsembleConsensusOut,
    JobCreate,
    JobOut,
    JobSummary,
    ReviewSegmentsOut,
    TempoMapOut,
)
from .tempo import TempoMapError, build_tempo_map


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="多轨录音节拍对齐 API",
    version="1.5.0",
    description=(
        "提交参考节拍网格与演奏轨音符起始时间，服务使用带宽约束的动态时间规整"
        "（Sakoe-Chiba DTW）处理漏拍、抢拍与插入拍，回溯生成一一对应映射及"
        "连续偏差区间。创建任务时可选提供 anchors 锚点（参考拍索引、演奏音索引），"
        "锚点作为硬约束把 DTW 限制在相邻锚点之间分段寻优，合并为完整对齐路径。"
        "已完成的任务可调用 /jobs/{id}/review-segments，按偏差阈值与可容忍拍间隔"
        "把漏拍、插入音及超阈值偏差聚合成复核片段（拍号范围 + 问题摘要）。"
        "多个共享同一参考节拍网格的已完成任务可调用 /ensembles/consensus："
        "先以各轨匹配拍偏差中位数估计并校正固定录音延迟，再按参考拍汇总，"
        "至少 3 轨匹配的拍取校正时间中位数为合奏基准并列出超阈值异常任务，"
        "覆盖不足的拍标记为不可判定且不报异常。"
        "调用 /takes/comp-plan 可对至少 2 条共享参考网格的已对齐录音按小节做"
        "动态规划拼轨：以完整匹配小节各拍的平均绝对偏差为选用代价、相邻小节"
        "换轨另付非负代价，总代价最小时优先请求顺序靠前的任务，返回合并连续"
        "同源小节的剪辑片段。"
    ),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.post("/jobs", response_model=JobOut, status_code=status.HTTP_201_CREATED, tags=["jobs"])
def create_job(payload: JobCreate, db: sqlite3.Connection = Depends(get_db)):
    """创建对齐任务（创建后状态为 pending，需调用 /align 执行对齐）。

    可提供 ``anchors`` 锚点硬约束：锚点必须严格递增、不越界且在带宽内可达，
    否则返回 400。
    """
    n, m = len(payload.reference_beats), len(payload.performance_onsets)
    if abs(n - m) > payload.params.band_width:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"序列长度差 {abs(n - m)} 超过带宽 {payload.params.band_width}，"
                "请增大 params.band_width"
            ),
        )
    anchor_pairs = (
        [(a.ref_index, a.perf_index) for a in payload.anchors]
        if payload.anchors
        else []
    )
    try:
        validate_anchors(anchor_pairs, n, m, payload.params.band_width)
    except AlignmentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        )
    return crud.create_job(
        db,
        name=payload.name,
        reference_beats=payload.reference_beats,
        performance_onsets=payload.performance_onsets,
        params=payload.params.model_dump(),
        anchors=[
            {"ref_index": a.ref_index, "perf_index": a.perf_index}
            for a in (payload.anchors or [])
        ],
    )


@app.post("/jobs/{job_id}/align", response_model=JobOut, tags=["jobs"])
def align_job(
    job_id: int,
    force: bool = Query(default=False, description="已完成时是否强制重新对齐"),
    db: sqlite3.Connection = Depends(get_db),
):
    """执行 DTW 对齐并持久化结果。"""
    job = crud.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if job["status"] == "done" and not force:
        return job

    params = AlignmentParams(
        band_width=job["params"].get("band_width", 3),
        skip_penalty=job["params"].get("skip_penalty"),
        insert_penalty=job["params"].get("insert_penalty"),
        on_time_threshold=job["params"].get("on_time_threshold", 0.03),
    )
    try:
        result = align_beats(
            job["reference_beats"],
            job["performance_onsets"],
            params,
            anchors=job.get("anchors") or [],
        )
    except AlignmentError as exc:
        crud.save_error(db, job_id, str(exc))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    crud.save_result(db, job_id, result)
    return crud.get_job(db, job_id)


@app.get("/jobs", response_model=list[JobSummary], tags=["jobs"])
def list_jobs(
    limit: int = Query(default=100, ge=1, le=500),
    db: sqlite3.Connection = Depends(get_db),
):
    """列出任务摘要。"""
    jobs = crud.list_jobs(db, limit)
    return [
        {
            "id": j["id"],
            "name": j["name"],
            "status": j["status"],
            "created_at": j["created_at"],
            "aligned_at": j["aligned_at"],
            "ref_beat_count": len(j["reference_beats"]),
            "perf_onset_count": len(j["performance_onsets"]),
        }
        for j in jobs
    ]


@app.get("/jobs/{job_id}", response_model=JobOut, tags=["jobs"])
def get_job(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    """查询单个任务（含完整对齐结果）。"""
    job = crud.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    return job


@app.get("/jobs/{job_id}/result", tags=["jobs"])
def get_job_result(job_id: int, db: sqlite3.Connection = Depends(get_db)):
    """只查询对齐结果；未对齐或失败时返回 409/400 提示。"""
    job = crud.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if job["status"] == "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="任务尚未对齐")
    if job["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"上次对齐失败：{job['error']}",
        )
    return {"job_id": job_id, "result": job["result"]}


@app.get(
    "/jobs/{job_id}/tempo-map",
    response_model=TempoMapOut,
    tags=["jobs"],
)
def get_job_tempo_map(
    job_id: int,
    window_beats: int = Query(
        default=5, ge=1, le=101,
        description="局部 BPM 的滑动中位数窗口半径（拍数，取中心±(window_beats//2)）",
    ),
    min_bpm: float = Query(default=40.0, gt=0, description="接受的最小 BPM"),
    max_bpm: float = Query(default=300.0, gt=0, description="接受的最大 BPM"),
    db: sqlite3.Connection = Depends(get_db),
):
    """从已完成任务的 matched_beats 导出节奏曲线（tempo map）。

    先按相邻匹配拍的参考拍跨度与演奏间隔估计局部 BPM，用滑动中位数窗口
    抑制漏拍 / 插入拍造成的异常间隔（半速 / 倍速），再把相对差不超过 2%
    的连续稳定区间压缩为按参考拍排序的 tempo events；异常间隔的起始参考拍
    索引在 ``ignored_ref_indices`` 中返回（不切断事件）。

    - 任务不存在：404
    - 任务尚未对齐：409；上次对齐失败：400
    - 匹配拍不足 2 个：409
    - ``min_bpm >= max_bpm`` 或参数越界：422
    """
    job = crud.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if job["status"] == "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="任务尚未对齐")
    if job["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"上次对齐失败：{job['error']}",
        )
    if min_bpm >= max_bpm:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="min_bpm 必须小于 max_bpm",
        )

    matched = (job["result"] or {}).get("matched_beats", [])
    try:
        tempo_map = build_tempo_map(
            matched,
            window_beats=window_beats,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
        )
    except TempoMapError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return {"job_id": job_id, **tempo_map}


@app.get(
    "/jobs/{job_id}/review-segments",
    response_model=ReviewSegmentsOut,
    tags=["jobs"],
)
def get_job_review_segments(
    job_id: int,
    deviation_threshold: float = Query(
        default=0.05, gt=0,
        description="偏差阈值（秒）：|偏差|严格大于该值的匹配拍列为超阈值偏差问题",
    ),
    gap_tolerance_beats: float = Query(
        default=2.0, ge=0,
        description=(
            "可容忍的拍间隔：相邻问题的最近参考拍索引差不超过该值时并入同一复核片段，"
            "否则另起片段（支持小数以容纳插入音的分数位置）"
        ),
    ),
    db: sqlite3.Connection = Depends(get_db),
):
    """从已对齐结果提取复核片段（按任务查询）。

    将漏拍、插入音及超阈值偏差按**参考拍顺序**聚合：相邻问题的最近参考拍
    索引差不超过 ``gap_tolerance_beats`` 时合并为一个复核片段；每个片段
    返回 1 基**拍号范围**（另附 0 基参考索引）、问题计数、逐问题明细与
    中文摘要。没有问题时返回空片段列表（仍是 200）。

    - 任务不存在：404
    - 任务尚未对齐 / 无对齐结果：409
    - 上次对齐失败：400
    - 参数非法（阈值非正等）：422
    """
    job = crud.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
    if job["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"上次对齐失败：{job['error']}",
        )
    if job["status"] == "pending" or job["result"] is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="任务尚无对齐结果，请先调用 /jobs/{job_id}/align 执行对齐",
        )
    try:
        review = extract_review_segments(
            job["result"],
            reference_beats=job["reference_beats"],
            deviation_threshold=deviation_threshold,
            gap_tolerance_beats=gap_tolerance_beats,
        )
    except ReviewError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    return {"job_id": job_id, **review}


@app.post(
    "/ensembles/consensus",
    response_model=EnsembleConsensusOut,
    status_code=status.HTTP_200_OK,
    tags=["ensembles"],
)
def ensemble_consensus(
    payload: ConsensusRequest,
    db: sqlite3.Connection = Depends(get_db),
):
    """跨任务合奏一致性分析（按需计算、只读，不落库）。

    提交**至少 3 个已完成**任务 ID（每任务一条演奏轨）与秒级异常阈值：

    - 重复 ID：400；任务不存在：404；任务未对齐（`pending`）/无对齐结果：409；
      上次对齐失败（`error`）：400；参考节拍网格不完全相同：422；
      阈值非正或任务数不足 3：422。

    对每条轨以匹配拍 ``perf_time − ref_time`` 的**中位数**估计固定录音延迟并
    校正演奏时间；再按参考拍汇总各轨校正时间，**至少 3 轨匹配**时取中位数为
    合奏基准，并列出偏离基准严格超过阈值的任务 ID 与偏差（early/late）。
    匹配轨不足 3 的拍标记为**不可判定**且不报异常；漏拍逐轨列出。
    """
    if len(set(payload.job_ids)) != len(payload.job_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="任务 ID 不可重复",
        )

    jobs = []
    for job_id in payload.job_ids:
        job = crud.get_job(db, job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务 {job_id} 不存在",
            )
        if job["status"] == "pending":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"任务 {job_id} 尚未对齐",
            )
        if job["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"任务 {job_id} 上次对齐失败：{job['error']}",
            )
        if job["result"] is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"任务 {job_id} 尚无对齐结果，"
                    "请先调用 /jobs/{id}/align 执行对齐"
                ),
            )
        jobs.append(job)

    reference_beats = jobs[0]["reference_beats"]
    for job in jobs[1:]:
        if job["reference_beats"] != reference_beats:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"任务 {jobs[0]['id']} 与任务 {job['id']} 的参考节拍网格"
                    "不完全相同，合奏一致性仅接受完全一致的参考节拍网格"
                ),
            )

    try:
        consensus = build_consensus(jobs, payload.deviation_threshold)
    except EnsembleError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    return consensus


@app.post(
    "/takes/comp-plan",
    response_model=CompPlanOut,
    status_code=status.HTTP_200_OK,
    tags=["takes"],
)
def takes_comp_plan(
    payload: CompPlanRequest,
    db: sqlite3.Connection = Depends(get_db),
):
    """多轨按小节拼轨的剪辑方案（按需计算、只读，不落库）。

    提交**至少 2 个已完成**任务 ID（每任务一条已对齐录音）、每小节拍数与
    非负换轨代价。按参考拍把共享网格划分为小节，只有**完整匹配该小节全部拍**
    的录音可选，选用代价为该小节匹配拍的**平均绝对偏差**；动态规划最小化
    「各小节选用代价 + 相邻小节换轨代价」之和，同代价时优先请求顺序靠前的
    任务。返回合并连续同源小节后的片段（任务 ID、参考拍索引范围、来源首末
    匹配音时间）。

    - 重复 ID：400；任务不存在：404；任务未对齐（`pending`）/无对齐结果：409；
      上次对齐失败（`error`）：400；参考节拍网格不完全相同：422；
      任务数不足 2、每小节拍数非法或换轨代价为负：422；
      某小节没有任何录音完整匹配其全部拍：422（明细含小节序号）。
    """
    if len(set(payload.job_ids)) != len(payload.job_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="任务 ID 不可重复",
        )

    jobs = []
    for job_id in payload.job_ids:
        job = crud.get_job(db, job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"任务 {job_id} 不存在",
            )
        if job["status"] == "pending":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"任务 {job_id} 尚未对齐",
            )
        if job["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"任务 {job_id} 上次对齐失败：{job['error']}",
            )
        if job["result"] is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"任务 {job_id} 尚无对齐结果，"
                    "请先调用 /jobs/{id}/align 执行对齐"
                ),
            )
        jobs.append(job)

    reference_beats = jobs[0]["reference_beats"]
    for job in jobs[1:]:
        if job["reference_beats"] != reference_beats:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"任务 {jobs[0]['id']} 与任务 {job['id']} 的参考节拍网格"
                    "不完全相同，剪辑方案仅接受完全一致的参考节拍网格"
                ),
            )

    try:
        plan = build_comp_plan(
            jobs,
            beats_per_bar=payload.beats_per_bar,
            switch_cost=payload.switch_cost,
        )
    except CompPlanError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    return plan
