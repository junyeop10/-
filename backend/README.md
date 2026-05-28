# AI 파일 분류 시스템 — 서버 실행 메뉴얼

## 한 줄 요약

| 상황 | 명령 |
|------|------|
| **매번 테스트할 때** | `cd backend` → `uvicorn main:app --reload` |
| **최초 1회만** | 의존성 설치 + `.env` 설정 + `python-multipart` 설치 |

`uvicorn` 한 줄만으로 되려면 **이미 설치·설정이 끝난 상태**이고, **`backend` 폴더에서** 실행해야 합니다.

---

## 1. 최초 1회 설정

```powershell
cd C:\-\backend

pip install -r requirements.txt
pip install python-multipart

copy .env.example .env
```

`.env` 파일을 열어 키·Ollama 설정을 맞춥니다.

```env
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:0.5b
MAX_CONCURRENT_LLM=5
LLM_MIN_CONFIDENCE=0.60
MAX_FILE_SIZE_MB=50
```

**로컬 Qwen (Stage 5):** [Ollama](https://ollama.com) 설치 후 `ollama pull qwen2.5:3b` 실행. 미실행 시 Claude API로 자동 폴백합니다.

```powershell
python test_qwen.py
python test_claude.py
```

> `.env`는 git에 올라가지 않습니다. `git add` 전에 `.gitignore`에 `.env`가 있는지 확인하세요.

---

## 2. 서버 실행 (이후 테스트마다)

```powershell
cd C:\-\backend
uvicorn main:app --reload
```

정상이면 터미널에 다음이 보입니다.

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

브라우저: **http://localhost:8000/docs** (Swagger UI)

### 포트가 이미 사용 중일 때 (`WinError 10013`)

다른 터미널에서 uvicorn이 이미 떠 있거나, 이전 프로세스가 남아 있을 수 있습니다.

```powershell
netstat -ano | findstr ":8000"
```

`LISTENING` 줄의 마지막 숫자(PID)를 종료:

```powershell
taskkill /PID <PID번호> /F
```

또는 다른 포트 사용:

```powershell
uvicorn main:app --reload --port 8001
```

→ http://localhost:8001/docs

### 서버 중지

터미널에서 `Ctrl + C`

---

## 3. API 테스트 순서

### 3-1. Swagger로 REST 테스트

1. http://localhost:8000/docs 접속
2. **`POST /upload`** — PDF 등 파일 1개 선택 후 Execute  
   - 응답의 `job_id` 복사
3. **`GET /result/{job_id}``** — `job_id` 붙여넣기 후 Execute  
   - `status: "completed"` 와 `results` 확인

### 3-2. WebSocket 진행 상태 (선택)

Swagger만으로는 WebSocket 테스트가 어렵습니다. **업로드 전에** 연결해야 진행 메시지를 받을 수 있습니다.

예: 브라우저 개발자 도구 콘솔

```javascript
const jobId = "여기에-job_id";
const ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`);
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

그다음 Swagger에서 `/upload` 실행.

### 3-3. 캐시 히트 확인

1. 같은 PDF를 `/upload`로 한 번 올림 → `job_id` A
2. **동일 파일**을 다시 `/upload` → `job_id` B
3. B의 WebSocket 또는 결과에서 캐시 관련 동작 확인 (`pre_stage` 단계 `status: "cached"`)

### 3-4. 사용자 수정 반영 (`POST /confirm/{job_id}`)

```json
{
  "corrections": [
    {
      "filename": "example.pdf",
      "user_category": "보고서",
      "folder_description": ""
    }
  }
}
```

---

## 4. 명령어 체크리스트

| 할 일 | 필요한가? |
|--------|-----------|
| `pip install -r requirements.txt` | 최초 1회 / 패키지 추가 시 |
| `pip install python-multipart` | 최초 1회 (파일 업로드 필수) |
| `.env` 설정 | 최초 1회 + API 키 변경 시 |
| `uvicorn main:app --reload` | **매번** (backend 폴더에서) |

**다른 명령은 평소 테스트에 필수 아님.**

---

## 5. 자주 나는 문제

| 증상 | 원인 | 해결 |
|------|------|------|
| `WinError 10013` | 8000 포트 점유 | 위 2절 참고 |
| `python-multipart` 오류 | 미설치 | `pip install python-multipart` |
| `results`가 비어 있음 | 텍스트 추출 실패 / 임베딩 오류 | PDF·docx로 테스트; NumPy 오류 시 룰·LLM만 동작 |
| WebSocket 메시지 없음 | 업로드 **후** 연결 | 업로드 **전** WebSocket 연결 |
| LLM 분류 안 됨 | API 키·Ollama 미실행 | `ANTHROPIC_API_KEY` 또는 `ollama serve` 확인 |
| `classify_method: llm_local` 없음 | Qwen 실패 후 Claude만 사용 | Ollama 모델 pull·`.env`의 `OLLAMA_MODEL` 확인 |
| Ollama 500 / `unable to allocate` | RAM 부족으로 모델 로딩 실패 | 다른 앱 종료 또는 `ollama pull qwen2.5:0.5b` 후 `.env`에 `OLLAMA_MODEL=qwen2.5:0.5b` |
| `requirements.txt` 없음 | `backend` 폴더 안에서 `backend\requirements.txt` 실행 | `cd backend` 후 `pip install -r requirements.txt` |

---

## 6. 팀 키워드 설정 (JSON)

룰 분류에 쓰는 키워드는 **`backend/config/keywords.json`** 에 있습니다.

| 파일 | 역할 |
|------|------|
| `config/keywords.json` | 팀이 수정하는 키워드 목록 |
| `config/loader.py` | 서버 시작 시 JSON을 읽어 `BASE_KEYWORDS`로 제공 |

**연결 방식:** 별도 API 호출 없음. `/upload` → Stage1·Stage3가 시작할 때 로드된 `BASE_KEYWORDS`를 자동 사용.

```text
keywords.json  →  loader.py (서버 시작 시 load)  →  stage3_rule (파일명) / stage1_evidence (본문)
```

팀에서 키워드를 바꾼 뒤:

1. `keywords.json` 저장 (Git PR 권장)
2. **uvicorn 재시작** (`--reload`면 파일 저장 시 자동 반영될 수 있음)
3. 샘플 파일로 `/upload` → `/result` 확인

카테고리 이름은 `최종본`, `보고서`, `발표자료` 등 **한글 키**를 유지하세요. (`stage3` 룰 분류와 매핑됨)

---

## 7. 프로젝트 구조 (참고)

```
backend/
├── main.py              # FastAPI 앱, 엔드포인트 4개
├── config/
│   ├── keywords.json    # 팀 키워드 (여기 수정)
│   └── loader.py        # JSON → BASE_KEYWORDS
├── pipeline/            # v2: extract→ocr→rule→embedding→llm→cluster→review
│   ├── stage5_llm_local.py   # Qwen (Ollama)
│   └── stage5_llm_claude.py    # Claude 폴백
├── models/schemas.py    # 데이터 모델
├── db/cache.py          # xxhash SQLite 캐시
├── uploads/             # 업로드 임시 저장 (git 제외)
├── cache.db             # 캐시 DB (git 제외)
├── .env                 # 비밀 설정 (git 제외)
└── .env.example         # 설정 예시
```

---

## 8. 엔드포인트 요약

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/upload` | 파일 업로드, 백그라운드 파이프라인 시작 |
| WebSocket | `/ws/{job_id}` | 진행 상태 실시간 수신 |
| GET | `/result/{job_id}` | 분류 결과 조회 |
| POST | `/confirm/{job_id}` | 사용자 수정·피드백 저장 |
