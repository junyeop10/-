"""
stage7_review.py — Stage 7: 결과·검토큐 최종 정리

[역할] 분류 성공 목록·검토큐·군집 결과를 API 응답 한 덩어리로 묶습니다.
[입력] results, review_queue, clusters
[출력] {"results", "review_queue", "clusters"}
[담당] 정윤서 (feature/stage7-review) — UI 연동 예정
"""

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
