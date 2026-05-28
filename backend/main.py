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


@app.on_event("startup")
async def startup() -> None:
    """Ollama 가용 여부를 서버 시작 시 1회만 확인."""
    import pipeline.stage3_classify as s3

    s3._ollama_available = await s3.check_ollama()
    print(f"Ollama 사용 가능: {s3._ollama_available}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# job_id별 WebSocket·결과·임베딩 (메모리, 서버 재시작 시 소멸)
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
    """
    파일별 분류 파이프라인 (v2).

    Pre → 추출 → (OCR) → 룰 → 임베딩·LLM → job 단위 군집·버전 정리 → 검토 큐 반영.
    진행 상태는 WebSocket `stage` 키로 broadcast (CONVENTIONS.md §10-6).
    """
    total = len(files_info)
    results: list[ClassifyResult] = []
    review_queue: list[dict] = []
    cluster_job_items: list[dict] = []  # HDBSCAN 입력 (임베딩 있는 파일만)
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

            # Pre: 확장자·용량 검사, xxhash 캐시 조회
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

            if pre_result["status"] == "cached":  # 동일 파일 재업로드 시 LLM 생략
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
                    "stage": "text_extract",
                    "progress": progress,
                    "current_file": filename,
                    "status": "running",
                },
            )
            # 텍스트 추출 → 실패 시 OCR 폴백
            extract_result = stage0_extract.run(file_bytes, filename, ext)

            if extract_result["status"] == "failed":
                await broadcast(
                    job_id,
                    {
                        "stage": "ocr_fallback",
                        "progress": progress,
                        "current_file": filename,
                        "status": "running",
                    },
                )
            extract_result = stage2_ocr.run(
                file_bytes, filename, ext, extract_result
            )

            if extract_result["status"] == "failed":  # OCR까지 실패 → 검토 큐
                review_queue.append(
                    {
                        "filename": filename,
                        "reason": extract_result.get("reason", "텍스트 추출 실패"),
                    }
                )
                await broadcast(
                    job_id,
                    {
                        "stage": "ocr_fallback",
                        "progress": progress,
                        "current_file": filename,
                        "status": "review_queue",
                    },
                )
                continue

            await broadcast(
                job_id,
                {
                    "stage": "evidence_package",
                    "progress": progress,
                    "current_file": filename,
                    "status": "running",
                },
            )
            # 파일명 룰로 확정되면 임베딩·LLM 생략
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
                        "stage": "evidence_package",
                        "progress": progress,
                        "current_file": filename,
                        "status": "filename_rule",
                    },
                )
                continue

            await broadcast(
                job_id,
                {
                    "stage": "evidence_package",
                    "progress": progress,
                    "current_file": filename,
                    "status": "running",
                },
            )
            # 증거 패키지(임베딩·키워드) 구성 후 LLM 분류 (Qwen → Claude 폴백)
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
                    "stage": "local_llm",
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
            else:  # LLM 실패·저신뢰
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
                    "stage": "external_api"
                    if result.classify_method == "claude_api"
                    else "local_llm",
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

    # job 전체: 유사 문서 군집 → 버전 그룹 → 결과 정리
    await broadcast(
        job_id,
        {
            "stage": "cluster_hdbscan",
            "progress": f"{total}/{total}",
            "current_file": "",
            "status": "running",
        },
    )
    clusters = stage6_cluster.run(cluster_job_items)

    await broadcast(
        job_id,
        {
            "stage": "version_organize",
            "progress": f"{total}/{total}",
            "current_file": "",
            "status": "running",
        },
    )

    version_groups = stage4_version.run(results)
    stage4_version.clear_embeddings()

    reviewed = stage7_review.run(results, review_queue, clusters)

    # 확정·학습은 POST /confirm 에서 처리 (여기서는 UI용 ready 이벤트만)
    await broadcast(
        job_id,
        {
            "stage": "feedback_learning",
            "progress": f"{total}/{total}",
            "current_file": "",
            "status": "ready",
        },
    )

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
            "stage": "folder_complete",
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
    """파일 저장 후 job_id 반환, 파이프라인은 백그라운드 실행."""
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
    """사용자 수정을 피드백 DB에 저장 (다음 분류 시 임베딩 유사도에 반영)."""
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
