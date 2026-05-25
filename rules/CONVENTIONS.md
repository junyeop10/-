# 프로젝트 컨벤션

> AI 기반 파일 분류 시스템 | 한양대학교 ERICA 데이터처리 기업 프로젝트
> 모든 팀원은 코드 작성 전 이 문서를 숙지할 것
>
> **GitHub를 처음 쓰는 팀원:** [`GITHUB_BEGINNER.md`](./GITHUB_BEGINNER.md) 먼저 읽기 (clone → 브랜치 → commit → PR)

---

## 1. 브랜치 전략

### 브랜치 구조
```
main
└── feature/backend-server     ← 서버 담당
└── feature/stage0-extract     ← Stage 0 텍스트 추출 담당
└── feature/stage1-evidence    ← Stage 1 증거 패키지 담당
└── feature/stage2-cluster     ← Stage 2 군집화 담당
└── feature/stage3-classify    ← Stage 3 분류 담당
└── feature/stage4-version     ← Stage 4 버전 정리 담당
└── feature/stage6-feedback    ← Stage 6 피드백 담당
└── feature/frontend           ← 프론트엔드 담당
└── hotfix/...                 ← 긴급 버그 수정 시에만
```

### 규칙
- `main` 브랜치에 **직접 push 금지**
- 작업은 반드시 본인 `feature/` 브랜치에서 진행
- 완료 후 GitHub에서 Pull Request 생성 → 팀원 1명 이상 리뷰 후 merge
- PR 제목 형식: `[Stage 0] 텍스트 추출기 구현` 처럼 담당 파트 명시

---

## 2. 커밋 메시지

### 형식
```
<type>: <내용>
```

### type 종류
| type | 사용 상황 |
|------|-----------|
| `feat` | 새 기능 추가 |
| `fix` | 버그 수정 |
| `refactor` | 동작 변화 없는 코드 개선 |
| `docs` | 문서, 주석 수정 |
| `test` | 테스트 코드 추가/수정 |
| `chore` | 패키지 설치, 설정 파일 변경 |

### 예시
```bash
git commit -m "feat: PDF 텍스트 추출 3구간 분할 로직 구현"
git commit -m "feat: Claude API 비동기 호출 및 Semaphore 제한 추가"
git commit -m "fix: xxhash 중복 감지 캐시 키 충돌 수정"
git commit -m "refactor: EvidencePackage 임베딩 계산 Lazy Singleton 적용"
git commit -m "chore: sentence-transformers 패키지 추가"
git commit -m "docs: Stage 3 분류 로직 주석 보완"
```

### 규칙
- 한국어 작성 권장
- 50자 이내로 핵심만 작성
- 과거형 금지 (`구현했다` ❌ → `구현` ✅)

---

## 3. 폴더 및 파일 구조

```
project-root/
├── rules/
│   └── CONVENTIONS.md        ← 이 파일
│
├── server/                   ← 서버 담당
│   ├── main.py
│   ├── config/
│   │   ├── keywords.json     ← 팀 키워드 설정 (Git 관리)
│   │   └── loader.py         ← JSON 로드
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── pre_stage.py
│   │   ├── stage0_extract.py
│   │   ├── stage1_evidence.py
│   │   ├── stage2_cluster.py
│   │   ├── stage3_classify.py
│   │   ├── stage4_version.py
│   │   └── stage6_feedback.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        ← 공용 데이터 구조 (전 파트 참고)
│   ├── db/
│   │   ├── __init__.py
│   │   └── cache.py
│   ├── docs/                 ← 회의 템플릿 등 팀 문서
│   ├── .env                  ← 깃 업로드 금지
│   ├── .env.example
│   ├── .gitignore
│   ├── requirements.txt
│   └── README.md             ← 서버 실행·테스트 메뉴얼
│
├── frontend/                 ← 프론트엔드 담당
│   └── ...
│
└── README.md
```

### 파일명 규칙
- Python 파일: `snake_case.py`
- 설정/문서 파일: `UPPER_CASE.md` 또는 `kebab-case.md`
- 새 파이프라인 모듈 추가 시: `stage{번호}_{역할}.py` 형식 유지

---

## 4. Python 코드 스타일

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
파이프라인 함수의 반환 딕셔너리에서 `status` 값은 아래로 통일한다.

| status | 의미 |
|--------|------|
| `"ok"` | 정상 처리 완료 |
| `"success"` | 텍스트 추출 성공 (Stage 0) |
| `"ocr_fallback"` | OCR 폴백으로 성공 (Stage 0) |
| `"cached"` | xxhash 캐시 히트 (Pre-stage) |
| `"failed"` | 처리 실패 → 검토 큐 이동 |
| `"review_queue"` | 유효성 검사 실패 → 검토 큐 이동 |

---

## 5. API 키 및 보안

- `.env` 파일은 **절대 깃에 올리지 않는다**
- API 키는 코드에 직접 작성 금지
- Claude API 호출은 **`stage3_classify.py` 한 곳에서만** 수행
- 다른 모듈에서 `anthropic` 라이브러리 직접 import 금지

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

## 6. 공용 데이터 구조 (`schemas.py`)

- `EvidencePackage`, `ClassifyResult`, `FeedbackLog`, `FinalizedDocument`, `Category` 는 **`models/schemas.py` 에서만 정의**
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

## 7. 의존성 관리

- 패키지 추가 시 반드시 `requirements.txt` 에 추가 후 커밋
- 버전은 고정하지 않고 최신 유지 (단, `anthropic` 라이브러리는 버전 고정 권장)
- 파일 업로드 API 사용 시 `python-multipart` 필요 (서버 담당자가 requirements에 포함 여부 확인)
- 설치 명령어:

```bash
cd server
pip install -r requirements.txt
```

---

## 8. 깃 관련 주의사항

### `.gitignore` 필수 포함 항목
```
.env
__pycache__/
*.pyc
*.pyo
cache.db
uploads/
*.egg-info/
.DS_Store
```

### push 전 체크리스트
- [ ] `feature/` 브랜치에서 작업했는가
- [ ] `.env` 파일이 스테이징에 포함되지 않았는가 (`git status` 확인)
- [ ] `requirements.txt` 업데이트했는가 (새 패키지 설치 시)
- [ ] 커밋 메시지 형식이 맞는가

---

## 9. 서버 실행 방법

```bash
cd server

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

## 10. 팀원별 담당 파트 및 인터페이스 계약

각 파트 담당자는 아래 인터페이스를 반드시 지켜야 서버가 각 모듈을 정상 호출할 수 있다.

| 파트 | 파일 | 메인 함수 | 반환 타입 |
|------|------|-----------|-----------|
| Pre-stage | `pipeline/pre_stage.py` | `run(file_bytes, filename, modified_at)` | `dict` |
| Stage 0 | `pipeline/stage0_extract.py` | `run(file_bytes, filename, ext)` | `dict` |
| Stage 1 | `pipeline/stage1_evidence.py` | `run(file_bytes, filename, ext, size_kb, modified_at, xxhash, extract_result)` | `EvidencePackage` |
| Stage 2 | `pipeline/stage2_cluster.py` | `run(results)` | `list[ClassifyResult]` (현재 pass-through) |
| Stage 3 | `pipeline/stage3_classify.py` | `async run(evidence, feedback_embeddings)` | `ClassifyResult` |
| Stage 4 | `pipeline/stage4_version.py` | `run(results)` | `list[dict]` |
| Stage 6 | `pipeline/stage6_feedback.py` | `save_feedback(log)` / `finalize_document(result)` | `None` / `FinalizedDocument` |

> 함수 시그니처(이름, 인자, 반환 타입)를 변경해야 할 경우 반드시 서버 담당자에게 먼저 공유할 것.

### 키워드·설정 파일 (팀 공동 관리)

| 파일 | 수정 주체 | 용도 |
|------|-----------|------|
| `server/config/keywords.json` | 팀 합의 후 PR | 룰 분류용 키워드 |
| `server/.env` | 서버 담당 (비공개) | API 키, 동시 호출 수 등 |

---

## 11. API 결과 구조 (프론트·운영 참고)

`GET /result/{job_id}` 응답 요약:

| 필드 | 의미 |
|------|------|
| `results` | 자동 분류 **완료**된 파일 (`/confirm` 대상) |
| `review_queue` | 분류 **실패·보류** 파일 (현재 `/confirm` 미지원) |
| `version_groups` | Stage 4 버전 그룹 |


