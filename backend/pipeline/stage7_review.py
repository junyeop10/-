"""Stage 7 — 최종 분류·검토 큐 정리 (MVP)."""

from models.schemas import ClassifyResult


def run(
    results: list[ClassifyResult],
    review_queue: list[dict],
    clusters: list[dict],
) -> dict:
    """
    LLM·군집 결과를 묶어 API 응답 형태로 반환합니다.

    정윤서 UI 브랜치에서 미리보기·수정 기능과 연동 예정.
    """
    return {
        "results": results,
        "review_queue": review_queue,
        "clusters": clusters,
    }
