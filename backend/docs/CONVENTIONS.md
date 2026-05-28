# 백엔드 구현 컨벤션

> FastAPI·`backend/pipeline/` 코드 작성 시 참고하는 **상세** 문서입니다.  
> 브랜치·담당·협업·파이프라인 요약은 **[팀 공통 `rules/CONVENTIONS.md`](../../rules/CONVENTIONS.md)** 를 먼저 읽으세요.

---

## 1. `backend/` 폴더 구조

```
backend/
├── main.py           # 파이프라인 연결·API
├── pipeline/         # Stage 모듈 (담당자별 파일)
├── models/schemas.py # 공용 데이터 구조 (변경 시 팀 공지)
├── config/keywords.json
└── docs/             # 백엔드 상세 문서 (이 파일)
```

### 파이프라인 파일 ↔ Stage

| Stage | 파일 |
|:-----:|------|
| Pre | `pre_stage.py` |
| 1 | `stage0_extract.py` |
| 2 | `stage2_ocr.py` |
| 3 | `stage3_rule.py` |
| 4 | `stage4_embedding.py` |
| 5 | `stage3_classify.py`, `stage5_llm_local.py`, `stage5_llm_claude.py` |
| 6 | `stage6_cluster.py` |
| 7 | `stage7_review.py` |

---

## 2. Python 코드 스타일

### 기본 원칙

- Python 3.10 이상 기준
- 들여쓰기: **스페이스 4칸** (탭 금지)
- 한 줄 최대 **100자**
- 파일 인코딩: **UTF-8**

### 네이밍

```python
# 변수, 함수: snake_case
file_hash = compute_hash(file_bytes)
def run(file_bytes: bytes, filename: str) -> dict:

# 클래스, Enum: PascalCase
class EvidencePackage:
class Category(str, Enum):

# 상수: UPPER_SNAKE_CASE
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.hwp'}
MAX_FILE_SIZE_MB = 50
```

### 타입 힌트 필수

```python
# ✅ 올바른 예
def run(file_bytes: bytes, filename: str, modified_at: float) -> dict:

# ❌ 잘못된 예
def run(file_bytes, filename, modified_at):
```

### 파이프라인 모듈 함수 이름 통일

모든 파이프라인 모듈의 메인 진입 함수는 `run()` 으로 통일한다.

```python
# 모든 stage 모듈 공통 형식
def run(...) -> dict:
    ...
```

### 예외 처리 필수

파이프라인 함수 내에서 예외가 발생해도 서버가 죽으면 안 된다.
반드시 `try/except` 로 감싸고 검토 큐로 보낼 것.

```python
# ✅ 올바른 예
def run(file_bytes: bytes, filename: str, ext: str) -> dict:
    try:
        # 처리 로직
        ...
    except Exception as e:
        return {"status": "failed", "reason": str(e)}

# ❌ 잘못된 예 — 예외 그냥 던지기
def run(file_bytes: bytes, filename: str, ext: str) -> dict:
    text = extract(file_bytes)  # 여기서 터지면 서버 전체 죽음
    ...
```

### 반환 status 값 통일

파이프라인 함수의 **반환 딕셔너리**에서 `status` 값은 아래로 통일한다.  
(`main.py` WebSocket 진행 이벤트의 `stage`/`status` 문자열은 UI 계약에 따라 별도 정의 가능)


| status           | 의미                       |
| ---------------- | ------------------------ |
| `"ok"`           | 정상 처리 완료                 |
| `"success"`      | 텍스트 추출 성공 (Stage 0)      |
| `"ocr_fallback"` | OCR 폴백으로 성공 (Stage 0)    |
| `"cached"`       | xxhash 캐시 히트 (Pre-stage) |
| `"failed"`       | 처리 실패 → 검토 큐 이동          |
| `"review_queue"` | 유효성 검사 실패 → 검토 큐 이동      |


---

## 3. API 키 및 보안

- `.env` 파일은 **절대 깃에 올리지 않는다**
- API 키는 코드에 직접 작성 금지
- **Claude API** 호출은 `pipeline/stage5_llm_claude.py` 한 곳에서만 수행
- 다른 모듈에서 `anthropic` 직접 import 금지
- **로컬 LLM (qwen/Ollama)** 은 `pipeline/stage5_llm_local.py` (이세연) 에서만 호출

```python
# ✅ 올바른 예 — .env에서 읽기
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

# ❌ 잘못된 예 — 코드에 직접 입력
api_key = "sk-ant-xxxxxxxxxxxx"
```

---

## 4. 공용 데이터 구조 (`schemas.py`)

- `EvidencePackage`, `ClassifyResult`, `FeedbackLog`, `FinalizedDocument`, `Category` 는 `**models/schemas.py` 에서만 정의**
- 각 모듈에서 사용 시 반드시 import해서 사용
- 구조 변경 시 팀원 전체에게 공유 후 수정

```python
# ✅ 올바른 import
from models.schemas import EvidencePackage, Category

# ❌ 각 파일에 중복 정의 금지
@dataclass
class EvidencePackage:  # stage1_evidence.py 안에 또 정의하는 것 금지
    ...
```

---

## 5. 의존성 관리

- 패키지 추가 시 반드시 `requirements.txt` 에 추가 후 커밋
- 버전은 고정하지 않고 최신 유지 (단, `anthropic` 라이브러리는 버전 고정 권장)
- 파일 업로드 API 사용 시 `python-multipart` 필요 (서버 담당자가 requirements에 포함 여부 확인)
- 설치 명령어:

```bash
cd backend
pip install -r requirements.txt
```

---

## 6. 서버 실행 방법

```bash
cd backend

# 최초 1회
pip install -r requirements.txt
pip install python-multipart   # 파일 업로드용
cp .env.example .env           # API 키 설정

# 개발 환경 (자동 재시작)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# API 문서 확인
http://localhost:8000/docs
```

---

## 7. 파이프라인 구현 상세

`main.py` `run_pipeline` 순서·담당·mermaid: [rules/CONVENTIONS.md §4](../../rules/CONVENTIONS.md#4-파이프라인-요약-플로우차트-v2)

### 7-1. Stage 5 LLM — 동작 원칙

- **로컬 LLM(qwen2.5 3B/7B, Ollama)** 을 기본으로 사용 (비용·프라이버시).
- 신뢰도 부족·복잡 케이스만 **Claude API** 호출.
- 입력: 추출 텍스트(1500×3), Stage4 임베딩, `EvidencePackage`(프롬프트 컨텍스트).
- 출력: `category`, `confidence`, `reason`, `keywords` (JSON).

### 7-2. 설정 파일

| 파일 | Stage | 용도 |
|------|:-----:|------|
| `config/keywords.json` | 3 | 룰·키워드 (파일명·본문 협의) |
| `config/embedding.json` (예정) | 4 | SBERT 모델명 등 |
| `config/cluster.json` (예정) | 6 | HDBSCAN 파라미터 |
| `.env` | 5 | `ANTHROPIC_API_KEY`, `MAX_CONCURRENT_LLM`, Ollama URL |

### 7-3. 현재 코드 vs v2 (갭)

| v2 Stage | 현재 backend |
|:--------:|-------------|
| 1 추출 | `stage0_extract.py` (이름만 다름) ✅ |
| 2 OCR | `stage2_ocr.py` 스텁 (실 OCR는 담당자 확장) |
| 3 룰 | `stage3_rule.py` (파일명 키워드) ✅ |
| 4 임베딩 | `stage4_embedding.py` → `stage1_evidence` 위임 ✅ |
| 5 LLM | `stage5_llm_claude` ✅ / `stage5_llm_local` ✅ (Qwen→Claude) |
| 6 군집 | `stage6_cluster.py` (HDBSCAN) ✅ |
| 7 검토 | `stage7_review.py` + API ✅ |
| 순서 | ✅ `main.py`: Pre→추출→(필요 시 OCR)→룰→임베딩 패키지→로컬 LLM→(필요 시 외부 API)→군집→버전→검토 |

### 7-4. WebSocket 진행 이벤트 키(`stage`)

프론트 진행 표시에서 사용하는 `stage` 값은 현재 아래와 같이 고정한다.

- `pre_stage`
- `text_extract`
- `ocr_fallback`
- `evidence_package`
- `local_llm`
- `external_api`
- `cluster_hdbscan`
- `version_organize`
- `feedback_learning`
- `folder_complete`


---

## 8. Stage 함수 계약

담당·브랜치 표: [rules/CONVENTIONS.md §5](../../rules/CONVENTIONS.md#5-팀원별-담당)


| Stage | 파일 | 메인 함수 | 반환 |
|:-----:|------|-----------|------|
| Pre | `pre_stage.py` | `run(file_bytes, filename, modified_at)` | `dict` |
| 1 | `stage0_extract.py` | `run(file_bytes, filename, ext)` | `dict` |
| 2 | `stage2_ocr.py` | `run(file_bytes, filename, ext)` | `dict` |
| 3 | `stage3_rule.py` | `run(filename, ext, xxhash, version_hint="")` | `ClassifyResult \| None` |
| 4 | `stage4_embedding.py` | `run(text_chunks, ...)` → `EvidencePackage` 조립 포함 가능 | `list[float]` / `EvidencePackage` |
| 5 | `stage5_llm_local.py` | `async classify_with_qwen(pkg)` | `dict` |
| 5 | `stage5_llm_claude.py` | `async classify_with_claude(pkg)` | `dict` |
| 6 | `stage6_cluster.py` | `run(job_embeddings)` | `list[dict]` 군집 메타 |
| 7 | `stage7_review.py` | `run(llm_results, clusters)` | `list[ClassifyResult]` |
| 8 | `stage8_feedback.py` | `save_feedback` … | — (MVP 제외) |


> 함수 시그니처 변경은 **반드시** `feature/backend-server` 담당자와 협의 후 `schemas.py` 와 함께 수정.

### 키워드·설정 파일 (팀 공동 관리)


| 파일                            | 수정 주체       | 용도               |
| ----------------------------- | ----------- | ---------------- |
| `backend/config/keywords.json` | 팀 합의 후 PR | ⚠️ 현재 본문용 — 룰용은 `keywords_filename.json` 분리 예정 |
| `backend/.env`                 | 서버 담당 (비공개) | API 키, 동시 호출 수 등 |


