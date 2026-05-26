# AI 파일 분류 시스템 — 코드 리뷰 스타일 가이드

이 저장소의 PR 리뷰 시 아래 규칙을 **반드시** 적용한다.  
상세: `rules/CONVENTIONS.md`, Git 워크플로: `rules/GITHUB_BEGINNER.md`

## Python

- Python 3.10+, 들여쓰기 스페이스 4칸, 한 줄 최대 100자
- 타입 힌트 필수 (`def run(...) -> dict:`)
- snake_case(함수·변수), PascalCase(클래스·Enum), UPPER_SNAKE_CASE(상수)

## 파이프라인 (`server/pipeline/`)

- 메인 진입 함수 이름은 **`run()`** 으로 통일
- 함수 시그니처 변경 시 `main.py` 호출부와 팀에 영향 — 리뷰에서 지적
- 예외는 삼키고 `status: "failed"` 또는 검토 큐로 반환 (서버 전체 크래시 금지)
- `status` 값: `ok`, `success`, `ocr_fallback`, `cached`, `failed`, `review_queue` 만 사용

## 데이터 구조

- `EvidencePackage`, `ClassifyResult`, `Category` 등은 **`server/models/schemas.py` 에만 정의**
- 다른 파일에 dataclass 중복 정의 금지

## 보안

- `.env`, API 키 하드코딩 금지
- `anthropic` 호출은 **`pipeline/stage5_llm_claude.py` 한 곳만**
- 로컬 LLM(qwen)은 **`pipeline/stage5_llm_local.py` 한 곳만**
- 파이프라인 순서: 플로우차트 v2 — Stage1~7 (`rules/CONVENTIONS.md` §10)

## 키워드·설정

- 팀 키워드는 `server/config/keywords.json` — `stage1_evidence.py`에 키워드 하드코딩 추가 지양

## Git (PR 관점)

- `main` 직접 push 가정 금지
- 커밋 메시지: `feat:`, `fix:`, `docs:` 등 + 한국어 50자 이내
- `.env`, `uploads/`, `cache.db` 커밋 포함 시 **반드시 BLOCK 코멘트**

## 리뷰 톤

- 문제점 + 수정 예시 코드 제안
- 사소한 스타일만 지적하지 말고, 위 규칙 위반·버그·인터페이스 불일치 우선
