import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from db import cache
from models.schemas import Category, ClassifyResult, EvidencePackage, FeedbackLog
from pipeline import (
    pre_stage,
    stage0_extract,
    stage2_ocr,
    stage3_classify,
    stage3_rule,
    stage4_embedding,
    stage4_version,
    stage6_cluster,
    stage6_feedback,
    stage7_review,
)

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="AI 파일 분류 시스템")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

_connections: dict[str, list[WebSocket]] = {}
_job_results: dict[str, dict] = {}
_job_embeddings: dict[str, dict[str, list[float]]] = {}


def _classify_result_to_dict(result: ClassifyResult) -> dict:
    d = asdict(result)
    d["category"] = result.category.value
    return d


def _version_group_to_dict(group: dict) -> dict:
    return {
        "representative": _classify_result_to_dict(group["representative"]),
        "versions": [_classify_result_to_dict(v) for v in group["versions"]],
        "is_duplicate": group["is_duplicate"],
    }


async def broadcast(job_id: str, message: dict) -> None:
    if job_id not in _connections:
        return
    dead: list[WebSocket] = []
    for ws in _connections[job_id]:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connections[job_id].remove(ws)


async def run_pipeline(job_id: str, files_info: list[dict]) -> None:
    total = len(files_info)
    results: list[ClassifyResult] = []
    review_queue: list[dict] = []
    cluster_job_items: list[dict] = []
    feedback_embeddings = stage6_feedback.get_feedback_embeddings()

    for idx, info in enumerate(files_info, start=1):
        file_path = info["path"]
        filename = info["filename"]
        modified_at = info["modified_at"]
        progress = f"{idx}/{total}"

        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()

            ext = Path(filename).suffix.lower()
            size_kb = len(file_bytes) / 1024

            await broadcast(
                job_id,
                {
                    "stage": "pre_stage",
                    "progress": progress,
                    "current_file": filename,
                    "status": "running",
                },
            )

            pre_result = pre_stage.run(file_bytes, filename, modified_at)

            if pre_result["status"] == "review_queue":
                review_queue.append(
                    {"filename": filename, "reason": pre_result["reason"]}
                )
                await broadcast(
                    job_id,
                    {
                        "stage": "pre_stage",
                        "progress": progress,
                        "current_file": filename,
                        "status": "review_queue",
                        "reason": pre_result["reason"],
                    },
                )
                continue

            if pre_result["status"] == "cached":
                cached = pre_result["cached_result"]
                try:
                    category = Category(cached["category"])
                except ValueError:
                    category = Category.UNCLASSIFIED
                result = ClassifyResult(
                    filename=filename,
                    file_path=file_path,
                    xxhash=pre_result.get("xxhash", ""),
                    category=category,
                    confidence=float(cached.get("confidence", 0)),
                    reason="캐시 히트",
                    keywords=[],
                    classify_method="rule",
                )
                results.append(result)
                await broadcast(
                    job_id,
                    {
                        "stage": "pre_stage",
                        "progress": progress,
                        "current_file": filename,
                        "status": "cached",
                    },
                )
                continue

            xxhash = pre_result["xxhash"]

            await broadcast(
                job_id,
                {
                    "stage": "stage1_extract",
                    "progress": progress,
                    "current_file": filename,
                    "status": "running",
                },
            )
            extract_result = stage0_extract.run(file_bytes, filename, ext)

            await broadcast(
                job_id,
                {
                    "stage": "stage2_ocr",
                    "progress": progress,
                    "current_file": filename,
                    "status": "running",
                },
            )
            extract_result = stage2_ocr.run(
                file_bytes, filename, ext, extract_result
            )

            if extract_result["status"] == "failed":
                review_queue.append(
                    {
                        "filename": filename,
                        "reason": extract_result.get("reason", "텍스트 추출 실패"),
                    }
                )
                await broadcast(
                    job_id,
                    {
                        "stage": "stage2_ocr",
                        "progress": progress,
                        "current_file": filename,
                        "status": "review_queue",
                    },
                )
                continue

            await broadcast(
                job_id,
                {
                    "stage": "stage3_rule",
                    "progress": progress,
                    "current_file": filename,
                    "status": "running",
                },
            )
            rule_result = stage3_rule.run(filename, ext, xxhash)
            if rule_result is not None:
                rule_result.file_path = file_path
                cache.set_cache(
                    xxhash, rule_result.category.value, rule_result.confidence
                )
                results.append(rule_result)
                await broadcast(
                    job_id,
                    {
                        "stage": "stage3_rule",
                        "progress": progress,
                        "current_file": filename,
                        "status": "rule",
                    },
                )
                continue

            await broadcast(
                job_id,
                {
                    "stage": "stage4_embedding",
                    "progress": progress,
                    "current_file": filename,
                    "status": "running",
                },
            )
            evidence = stage4_embedding.run(
                file_bytes, filename, ext, size_kb, modified_at, xxhash, extract_result
            )
            stage4_version.register_embedding(xxhash, evidence.embedding)
            _job_embeddings.setdefault(job_id, {})[xxhash] = evidence.embedding
            if evidence.embedding:
                cluster_job_items.append(
                    {
                        "xxhash": xxhash,
                        "embedding": evidence.embedding,
                        "filename": filename,
                    }
                )

            await broadcast(
                job_id,
                {
                    "stage": "stage5_llm",
                    "progress": progress,
                    "current_file": filename,
                    "status": "running",
                },
            )
            result = await stage3_classify.run(evidence, feedback_embeddings)
            result.file_path = file_path

            if result.classify_method != "review_queue":
                cache.set_cache(
                    xxhash, result.category.value, result.confidence
                )
                results.append(result)
            else:
                review_queue.append(
                    {
                        "filename": filename,
                        "reason": result.review_reason or result.reason,
                        "xxhash": xxhash,
                    }
                )

            await broadcast(
                job_id,
                {
                    "stage": "stage5_llm",
                    "progress": progress,
                    "current_file": filename,
                    "status": result.classify_method,
                },
            )

        except Exception as e:
            review_queue.append({"filename": filename, "reason": str(e)})
            await broadcast(
                job_id,
                {
                    "stage": "error",
                    "progress": progress,
                    "current_file": filename,
                    "status": "review_queue",
                    "reason": str(e),
                },
            )

    await broadcast(
        job_id,
        {
            "stage": "stage6_cluster",
            "progress": f"{total}/{total}",
            "current_file": "",
            "status": "running",
        },
    )
    clusters = stage6_cluster.run(cluster_job_items)

    await broadcast(
        job_id,
        {
            "stage": "stage4_version",
            "progress": f"{total}/{total}",
            "current_file": "",
            "status": "running",
        },
    )

    version_groups = stage4_version.run(results)
    stage4_version.clear_embeddings()

    reviewed = stage7_review.run(results, review_queue, clusters)

    _job_results[job_id] = {
        "status": "completed",
        "results": reviewed["results"],
        "version_groups": version_groups,
        "review_queue": reviewed["review_queue"],
        "clusters": reviewed["clusters"],
    }

    await broadcast(
        job_id,
        {
            "stage": "complete",
            "progress": f"{total}/{total}",
            "current_file": "",
            "status": "completed",
        },
    )


@app.post(
    "/upload",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            }
                        },
                    }
                }
            }
        }
    },
)
async def upload(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File()],
):
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    files_info = []
    for upload in files:
        filename = upload.filename or "unknown"
        dest = job_dir / filename
        content = await upload.read()
        with open(dest, "wb") as f:
            f.write(content)
        files_info.append(
            {
                "path": str(dest),
                "filename": filename,
                "modified_at": time.time(),
            }
        )

    _job_results[job_id] = {
        "status": "processing",
        "results": [],
        "version_groups": [],
        "review_queue": [],
        "clusters": [],
    }
    background_tasks.add_task(run_pipeline, job_id, files_info)

    return {"job_id": job_id, "file_count": len(files_info)}


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    _connections.setdefault(job_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in _connections and websocket in _connections[job_id]:
            _connections[job_id].remove(websocket)


@app.post("/confirm/{job_id}")
async def confirm(job_id: str, body: dict):
    corrections = body.get("corrections", [])
    job_data = _job_results.get(job_id, {})
    if not job_data:
        return {
            "saved": 0,
            "details": [],
            "available_filenames": [],
            "hint": "job_id를 찾을 수 없습니다. 서버 재시작 후에는 /upload부터 다시 하세요.",
        }

    results_by_name = {
        r.filename: r for r in job_data.get("results", [])
    }
    review_names = [r.get("filename") for r in job_data.get("review_queue", [])]

    saved = 0
    details: list[dict] = []
    for correction in corrections:
        filename = correction.get("filename", "")
        user_cat_str = correction.get("user_category", "")
        result = results_by_name.get(filename)
        if not result:
            hint = (
                "검토 큐에만 있습니다. 분류가 완료된 파일만 confirm 가능합니다."
                if filename in review_names
                else "results에 없는 파일명입니다. GET /result 의 filename과 정확히 일치해야 합니다."
            )
            details.append({"filename": filename, "status": "skipped", "reason": hint})
            continue

        try:
            user_category = Category(user_cat_str)
        except ValueError:
            details.append(
                {
                    "filename": filename,
                    "status": "skipped",
                    "reason": f"잘못된 user_category: {user_cat_str}",
                }
            )
            continue

        system_category = result.category
        corrected = user_category != system_category

        embedding = _job_embeddings.get(job_id, {}).get(result.xxhash, [])

        log = FeedbackLog(
            xxhash=result.xxhash,
            embedding=embedding,
            system_category=system_category,
            user_category=user_category,
            corrected=corrected,
            correction_stage="user_confirm",
            timestamp=time.time(),
        )
        stage6_feedback.save_feedback(log)

        if user_category == Category.FINAL:
            stage6_feedback.finalize_document(result)

        saved += 1
        details.append({"filename": filename, "status": "saved"})

    return {
        "saved": saved,
        "details": details,
        "available_filenames": list(results_by_name.keys()),
    }


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    job_data = _job_results.get(job_id)
    if not job_data:
        return {
            "status": "not_found",
            "results": [],
            "version_groups": [],
            "review_queue": [],
            "clusters": [],
        }

    return {
        "status": job_data.get("status", "unknown"),
        "results": [
            _classify_result_to_dict(r) for r in job_data.get("results", [])
        ],
        "version_groups": [
            _version_group_to_dict(g) for g in job_data.get("version_groups", [])
        ],
        "review_queue": job_data.get("review_queue", []),
        "clusters": job_data.get("clusters", []),
    }
