# Claude LLM 분류 테스트 메뉴얼

이 패키지는 **파일 업로드 서버 없이** 로컬 파일로 Claude API 분류만 검증합니다.

## 포함된 핵심 코드

| 파일 | 역할 |
|------|------|
| `pipeline/stage5_claude.py` | Claude API 호출 |
| `pipeline/stage5_common.py` | 시스템/유저 프롬프트, JSON 파싱 |
| `test_claude.py` | 테스트 실행 스크립트 |

## 사전 요구사항

- **Python 3.10 이상**
- **Anthropic API 키** — [console.anthropic.com](https://console.anthropic.com) 에서 발급
- 테스트용 샘플 파일 1~2개 (PDF, DOCX 권장)

## 1. 최초 1회 설정

```powershell
cd claude-llm-test

pip install -r requirements.txt

copy .env.example .env
```

`.env` 파일을 열어 API 키를 입력합니다.

```env
ANTHROPIC_API_KEY=sk-ant-여기에입력
MAX_CONCURRENT_LLM=5
LLM_MIN_CONFIDENCE=0.60
MAX_FILE_SIZE_MB=50
```

> **주의:** `.env`는 절대 zip이나 Git에 공유하지 마세요. API 키는 Slack DM 등 별도 채널로 전달하세요.

첫 실행 시 `sentence-transformers` 모델(약 400MB)이 자동 다운로드됩니다. 네트워크가 필요하며 1회만 발생합니다.

## 2. 테스트 명령

```powershell
# 1) API 호출 없이 프롬프트만 확인 (비용 0원)
python test_claude.py C:\samples\보고서.pdf --dry-run

# 2) 실제 Claude 분류 (API 호출)
python test_claude.py C:\samples\보고서.pdf

# 3) 여러 파일
python test_claude.py file1.docx file2.pdf

# 4) 폴더 안의 지원 확장자 전부
python test_claude.py C:\samples\

# 5) 본문 추출 실패해도 강제로 LLM 호출
python test_claude.py broken.pdf --force-llm
```

### 지원 확장자

`.pdf` `.docx` `.hwp` `.hwpx` `.pptx` `.xlsx` `.jpg` `.jpeg` `.png`

## 3. 정상 출력 예시

```json
{
  "ok": [
    {
      "file": "C:\\samples\\보고서.pdf",
      "filename": "보고서.pdf",
      "extract_status": "success",
      "extract_method": "pymupdf",
      "text_chars": 4521,
      "keyword_hits": ["보고서", "분석"],
      "version_hint": "",
      "text_front_preview": "2024년 분기 보고서 ...",
      "llm": {
        "category": "보고서",
        "confidence": 0.87,
        "reason": "본문에 분석 결과와 현황이 포함됨",
        "keywords": ["보고서", "분석", "결과"]
      }
    }
  ],
  "errors": []
}
```

## 4. 확인 포인트 (피드백 요청)

테스트 후 아래 항목을 팀 채널에 공유해 주세요.

1. **category** 가 아래 7종 중 하나인지
   - `최종본`, `발표자료`, `보고서`, `데이터`, `참고자료`, `작업중`, `분류불가`
2. **confidence** 가 0.0 ~ 1.0 사이인지
3. **reason**, **keywords** 가 한국어인지
4. 파일명만 있는 경우 vs 본문이 있는 경우 분류 차이
5. **잘못 분류된 케이스** — 파일명, 기대 카테고리, 실제 결과

## 5. 동작 흐름

```
로컬 파일
  → 텍스트 추출 (PDF/DOCX 등)
  → 키워드·임베딩으로 EvidencePackage 구성
  → Claude API (claude-sonnet-4-20250514)
  → JSON 파싱 → category / confidence / reason / keywords
```

- API 오류·JSON 파싱 실패 시 **1회 재시도**
- 그래도 실패하면 `category: "분류불가"`, `confidence: 0.0`

## 6. 자주 나는 문제

| 증상 | 원인 | 해결 |
|------|------|------|
| `skipped: ANTHROPIC_API_KEY 없음` | `.env` 미설정 | `.env`에 키 입력 |
| `llm_skipped: true` | 본문 추출 실패 | 다른 파일로 재시도 또는 `--force-llm` |
| 첫 실행이 매우 느림 | SBERT 모델 다운로드 | 1회 대기 (약 400MB) |
| NumPy DLL 오류 (Windows) | 버전 충돌 | `pip install -r requirements.txt` 재실행 |
| `llm.reason: API 오류` | 키·잔액·네트워크 | Anthropic 콘솔 확인 |
| `분류불가` + confidence 0 | 파싱 실패 | 같은 파일 재시도, 계속되면 이슈 보고 |

## 7. 키워드 수정 (선택)

`config/keywords.json` 을 수정하면 본문 키워드 매칭 결과가 바뀝니다.  
Claude 프롬프트의 `keyword_hits` 필드에 반영됩니다.

수정 후 스크립트를 다시 실행하면 됩니다 (서버 재시작 불필요).

## 8. 문의 시 첨부할 것

- 테스트한 **파일명** (민감하면 익명화)
- `test_claude.py` **전체 JSON 출력**
- 사용한 **OS** (Windows / Mac)
- `.env`에 키가 있는지 여부 (키 값 자체는 공유 금지)
