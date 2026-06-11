# Claude LLM 분류 테스트 메뉴얼

이 패키지는 **파일 업로드 서버 없이** 로컬 파일로 Claude API 분류만 검증합니다.

## 기본 카테고리 (8개)

| # | 카테고리 |
|---|----------|
| 1 | 공고_지침_양식 |
| 2 | 사업계획서 수행계획서 |
| 3 | 조사_참고자료 |
| 4 | 중간_최종 결과물 및 보고서 |
| 5 | 발표자료 |
| 6 | 견적_계약_정산 |
| 7 | 기업 인증서 |
| 8 | 기타 |

> `분류불가`는 API 오류·검토 큐용 **시스템 상태**이며 LLM이 고르는 8개 카테고리에 포함되지 않습니다.

## 포함된 핵심 코드

| 파일 | 역할 |
|------|------|
| `pipeline/stage5_claude.py` | Claude API 1차·RAG 2차 호출 |
| `pipeline/stage5_common.py` | 프롬프트·JSON 파싱 |
| `pipeline/stage5_rag.py` | RAG 유사 카테고리 힌트 (DB 연동 전 스텁) |
| `config/categories.json` | 카테고리 설명·예시 (RAG용) |
| `test_claude.py` | 테스트 실행 스크립트 |

## 분류 흐름 (서버·통합 기준)

```
1차 Claude 분류
  → 확신 있으면 완료
  → 저신뢰·기타·실패 시 RAG 힌트 검색
  → 2차 Claude (RAG 프롬프트)
  → 기존 카테고리 매칭 / 새 카테고리 제안(검토큐)
```

`test_claude.py`는 **1차 Claude만** 호출합니다. RAG 2차는 서버(`uvicorn`) 또는 `stage5_classify.run()`에서 동작합니다.

## 사전 요구사항

- **Python 3.10 이상**
- **Anthropic API 키**
- 테스트용 샘플 파일 1~2개 (PDF, DOCX 권장)

## 1. 최초 1회 설정

```powershell
cd claude-llm-test

pip install -r requirements.txt

copy .env.example .env
```

```env
ANTHROPIC_API_KEY=sk-ant-여기에입력
MAX_CONCURRENT_LLM=5
LLM_MIN_CONFIDENCE=0.60
MAX_FILE_SIZE_MB=50
```

## 2. 테스트 명령

```powershell
python test_claude.py C:\samples\직무수행계획서.pdf --dry-run
python test_claude.py C:\samples\직무수행계획서.pdf
python test_claude.py C:\samples\
python test_claude.py broken.pdf --force-llm
```

## 3. 정상 출력 예시

```json
{
  "ok": [{
    "filename": "직무수행계획서 200615.pdf",
    "extract_status": "success",
    "llm": {
      "category": "사업계획서 수행계획서",
      "confidence": 0.87,
      "reason": "사업 수행 계획과 컨설팅 일정이 포함됨",
      "keywords": ["수행계획", "컨설팅"]
    }
  }],
  "errors": []
}
```

## 4. RAG 2차 분류 (서버)

`GET /result/{job_id}` 에서 확인:

- `classify_method: "claude_rag"` — RAG 보강 후 분류 성공
- `review_queue` + `is_new_category: true` — 새 카테고리 제안 (사람 검토 필요)
- `reason: "크레딧 부족"` 등 — API 문제

## 5. 자주 나는 문제

| 증상 | 해결 |
|------|------|
| `크레딧 부족` | Anthropic 콘솔에서 크레딧 충전 |
| `API 키 미설정` | `.env` 확인 |
| `review_queue`만 나옴 | 크레딧·추출 상태·`LLM_MIN_CONFIDENCE` 확인 |

## 6. 키워드·카테고리 수정

- `config/keywords.json` — 본문·파일명 키워드
- `config/categories.json` — RAG 설명·예시 파일명

수정 후 스크립트/서버 재실행.
