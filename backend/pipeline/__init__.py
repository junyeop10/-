"""
backend/pipeline — 파일 분류 파이프라인 모듈 모음

각 stage*.py 는 `run()` 함수 하나로 진입합니다. main.py 가 아래 순서로 호출합니다.

  pre_stage        → 확장자·용량 검사, xxhash 캐시
  stage0_extract   → PDF/DOCX 등 본문 텍스트 추출
  stage2_ocr       → 추출 실패 시 OCR (현재 스텁)
  stage3_rule      → 파일명 키워드 룰 분류 (확정 시 이후 단계 생략)
  stage4_embedding → 임베딩·키워드·EvidencePackage 구성
  stage5_classify  → 임베딩 유사도 + Claude 1차 + RAG 2차
  stage6_cluster   → HDBSCAN 군집 (회의 결정: 추후 제거 예정)
  stage4_version   → 버전·중복 그룹 정리
  stage7_review    → 결과·검토큐 정리
  stage6_feedback  → 사용자 수정 로그 저장 (/confirm)
"""
