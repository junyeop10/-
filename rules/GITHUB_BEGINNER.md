# GitHub 초보자 가이드 — AI 파일 분류 프로젝트

> Git을 **처음** 쓰는 팀원용입니다.  
> 개발 용어는 최소화하고, **복사해서 붙여넣을 명령어** 위주로 정리했습니다.  
> 브랜치·커밋 규칙은 [`CONVENTIONS.md`](./CONVENTIONS.md) 와 함께 보면 됩니다.

---

## 0. 이 문서로 할 수 있는 것

| 할 일 | 이 가이드 |
|--------|------------|
| GitHub 계정 만들기 | §1 |
| 프로젝트 코드 받기 (clone) | §2 |
| 내 담당 브랜치에서 작업하기 | §3~§5 |
| 저장·업로드 (commit / push) | §6 |
| 팀에 반영 요청 (Pull Request) | §7 |
| 문제 생겼을 때 | §8 |

---

## 1. 준비 (최초 1회)

### 1-1. GitHub 계정

1. https://github.com 접속 → **Sign up**
2. 이메일 인증 완료
3. 팀 리더에게 **저장소 초대** 받기 (메일 또는 GitHub 알림 → Accept)

### 1-2. Git 설치 (PC에 Git 없을 때)

1. https://git-scm.com/download/win 에서 Windows용 설치
2. 설치 시 기본 옵션으로 Next (PATH 추가되는 옵션 유지)
3. 설치 확인 — **PowerShell** 또는 **터미널** 열고:

```powershell
git --version
```

`git version 2.x.x` 처럼 나오면 성공.

### 1-3. GitHub 로그인 연결 (최초 1회)

저장소 주소 예시: `https://github.com/junyeop10/-.git`  
(팀에서 알려준 **실제 저장소 URL**로 바꿀 것)

**방법 A — GitHub Desktop (GUI, 추천 초보자)**

1. https://desktop.github.com 설치
2. GitHub 계정으로 로그인
3. **File → Clone repository** → 팀 저장소 선택 → 로컬 폴더 선택 (예: `C:\projects\file-classify`)

**방법 B — 명령어 (이 가이드 메인)**

아래 §2 clone 참고.

---

## 2. 프로젝트 코드 받기 (clone)

### 2-1. 폴더 정하기

예: `C:\projects` 에 프로젝트를 둘 것.

```powershell
cd C:\projects
```

### 2-2. clone (처음 한 번만)

```powershell
git clone https://github.com/팀계정/저장소이름.git
cd 저장소이름
```

clone 후 폴더 안에 `server/`, `rules/` 등이 보이면 성공.

### 2-3. 이미 clone 해 둔 사람 — 최신 받기

다른 팀원이 merge한 내용을 받을 때:

```powershell
cd C:\projects\저장소이름
git checkout main
git pull origin main
```

---

## 3. 브랜치 이해 (3분)

| 이름 | 비유 | 우리 프로젝트 |
|------|------|----------------|
| `main` | **완성본** 공용 책 | 직접 수정·push **금지** |
| `feature/내파트` | **내 초안 노트** | 여기서만 작업 |

담당 예시 (`CONVENTIONS.md` §1 · 플로우차트 v2):

| 담당 | 브랜치 | Stage |
|------|--------|:-----:|
| 서버 통합 | `feature/backend-server` | Pre |
| 김준엽 | `feature/frontend-upload`, `feature/stage1-extract` | 업로드, 1 |
| 정건우 | `feature/stage2-ocr`, `feature/stage3-rule` | 2, 3 |
| 천승원 | `feature/stage4-embedding`, `feature/stage6-cluster` | 4, 6 |
| 이세연 | `feature/stage5-llm-local`, `feature/stage5-llm-claude` | 5 |
| 정윤서 | `feature/stage7-review` | 7 |

**본인 브랜치가 뭔지** 팀 리더에게 꼭 확인.

---

## 4. 내 브랜치로 이동

### 4-1. 브랜치가 이미 있을 때 (원격에 있음)

```powershell
cd C:\projects\저장소이름
git fetch origin
git checkout feature/stage0-extract
```

`feature/stage0-extract` → **본인 브랜치 이름**으로 변경.

### 4-2. 내 브랜치가 아직 없을 때 (새로 만들기)

```powershell
git checkout main
git pull origin main
git checkout -b feature/stage0-extract
git push -u origin feature/stage0-extract
```

---

## 5. 작업 전 항상 확인

```powershell
git branch
```

`* feature/stage0-extract` 처럼 **별표가 본인 브랜치**에 있어야 함.  
`main`에 별표가 있으면 §4로 다시 이동.

```powershell
git status
```

- 빨간/수정된 파일 목록 확인
- **`.env` 가 목록에 있으면 안 됨** (§6-2 참고)

---

## 6. 저장하고 GitHub에 올리기 (commit & push)

### 6-1. 작업 흐름 (매번)

```
파일 수정 (VS Code / Cursor 등)
    → git add
    → git commit
    → git push
```

### 6-2. add 전에 꼭 확인

**절대 올리면 안 되는 것**

- `server/.env` (API 키)
- `server/uploads/`
- `server/cache.db`
- `__pycache__/`

```powershell
git status
```

`.env`가 보이면:

```powershell
git restore --staged server/.env
```

또는 add 하지 말 것. (`server/.gitignore`에 `.env` 있음)

### 6-3. 명령어 예시

```powershell
# server 폴더만 수정했을 때
git add server/pipeline/stage0_extract.py
git add server/requirements.txt

# 또는 server 전체 ( .env 는 gitignore 로 제외됨 )
git add server/

# rules 문서만
git add rules/
```

커밋 (메시지 형식은 CONVENTIONS.md):

```powershell
git commit -m "feat: PDF 텍스트 추출 3구간 분할 로직 구현"
```

GitHub에 업로드:

```powershell
git push origin feature/stage0-extract
```

`feature/stage0-extract` → 본인 브랜치 이름.

---

## 7. Pull Request (PR) — 팀에 “합쳐 주세요” 요청

`main`에 직접 push 하지 않고, **PR**로 합칩니다.

### 7-1. 웹에서 PR 만들기 (가장 쉬움)

1. https://github.com/팀계정/저장소 → **Pull requests**
2. **New pull request**
3. **base:** `main` ← **compare:** `feature/본인브랜치`
4. 제목 예: `[Stage 0] PDF 텍스트 추출 구현`
5. 변경 내용 간단히 적기 (무엇을, 왜)
6. **Create pull request**
7. 팀원 1명 이상 **Review** → 승인 후 **Merge pull request**

### 7-2. PR 전 스스로 체크

- [ ] `feature/` 브랜치에서 push 했는가
- [ ] `.env` 안 올렸는가
- [ ] `requirements.txt` 바꿨으면 같이 commit 했는가
- [ ] 커밋 메시지 `feat:` / `fix:` 형식인가

### 7-3. merge 후 내 PC 최신화

```powershell
git checkout main
git pull origin main
git checkout feature/본인브랜치
git merge main
```

충돌 나면 §8-3 또는 팀에 도움 요청.

---

## 8. 자주 하는 실수 & 해결

### 8-1. `git push` 거절됨 (rejected)

다른 사람이 먼저 같은 브랜치에 push한 경우:

```powershell
git pull origin feature/본인브랜치
# 충돌 없으면
git push origin feature/본인브랜치
```

### 8-2. 잘못된 브랜치에 commit 함

아직 push 안 했을 때:

```powershell
git log -1
# 마지막 커밋만 브랜치 옮기기 (팀원/리더에게 물어보고 진행 권장)
```

이미 push 했으면 **팀 리더에게** 알리기.

### 8-3. merge 충돌 (Conflict)

같은 파일 같은 줄을 둘이 수정했을 때 발생.

1. `git status` 로 충돌 파일 확인
2. VS Code/Cursor가 `<<<<<<<` 표시 → **어느 쪽 유지할지** 팀과 상의 후 수정
3. 수정 후:

```powershell
git add .
git commit -m "fix: merge conflict 해결"
git push
```

### 8-4. PowerShell에서 `&&` 안 됨

```powershell
# ❌
cd server && uvicorn main:app

# ✅
cd server
uvicorn main:app --reload
```

### 8-5. 한길 파일명·경로

가능하면 영문 경로 권장 (`C:\projects\...`). 한글 경로도 되지만 Git 오류 시 경로 의심.

---

## 9. 담당별 “첫 주” 할 일 체크리스트

### 공통 (모든 팀원)

- [ ] GitHub 계정 + 저장소 초대 수락
- [ ] Git 설치 + `git --version` 확인
- [ ] clone 완료
- [ ] `CONVENTIONS.md` 읽기
- [ ] 본인 `feature/...` 브랜치 checkout
- [ ] 테스트 commit 1개 + push + PR 연습 (문서 한 줄 수정도 OK)

### 파이프라인 담당 (pipeline 수정)

- [ ] `rules/CONVENTIONS.md` §10 파이프라인 순서·본인 브랜치 확인
- [ ] `server/models/schemas.py` 입출력 구조 확인
- [ ] **본인 담당 파일만** 수정 (예: 이세연 → `stage3_llm_claude.py` 만)
- [ ] `run()` / `classify_with_*` 시그니처 변경 시 **서버 담당자에게 먼저** 공지
- [ ] `cd server` → `uvicorn main:app --reload` 로 동작 확인

### 프론트엔드 담당

- [ ] `feature/frontend` 브랜치
- [ ] API 주소: `http://localhost:8000` (`/docs` 참고)

### 서버 담당

- [ ] `main.py` · `main` merge 권한 조율
- [ ] 다른 팀원 PR 리뷰

---

## 10. GitHub Desktop만 쓸 때 (명령어 대신)

| 하고 싶은 일 | Desktop 메뉴 |
|--------------|----------------|
| clone | File → Clone repository |
| 브랜치 변경 | Current branch → 선택 |
| 저장 | 왼쪽 Changes → Summary 입력 → **Commit**
| 업로드 | **Push origin**
| PR | Branch → **Create pull request** (브라우저 열림) |
| 최신 받기 | **Fetch origin** → **Pull origin** |

커밋 메시지는 Desktop의 Summary에 `feat: ...` 형식으로 입력.

---

## 11. 용어 사전 (30초)

| 용어 | 뜻 |
|------|-----|
| **repository (repo)** | 프로젝트 통째로 있는 GitHub 폴더 |
| **clone** | repo를 내 PC로 복사 |
| **branch** | parallel 작업 줄기 (`main`, `feature/...`) |
| **commit** | 저장 시점 스냅샷 |
| **push** | 내 commit을 GitHub에 업로드 |
| **pull** | GitHub 변경분을 내 PC로 받기 |
| **PR** | “내 브랜치를 main에 합쳐 주세요” 요청 |
| **merge** | PR 승인 후 main에 반영 |

---

## 12. 도움 요청할 때 붙여넣을 정보

팀 채팅에 아래 4가지를내면 해결이 빠릅니다.

```powershell
git --version
git branch
git status
git log -3 --oneline
```

+ 에러 메시지 **전체** 스크린샷 또는 복사.

---

## 13. 관련 문서

| 문서 | 내용 |
|------|------|
| [`rules/CONVENTIONS.md`](./CONVENTIONS.md) | 브랜치, 커밋, 코드 스타일 |
| [`server/README.md`](../server/README.md) | 서버 실행·API 테스트 |
| [`server/docs/team-meeting-template.md`](../server/docs/team-meeting-template.md) | 키워드·룰 회의 템플릿 |

---

## 14. 저장소 URL만 바꿔 쓰는 치트시트

```powershell
# === 설정 (본인 환경에 맞게 한 번만) ===
cd C:\projects\저장소이름
git checkout feature/본인브랜치

# === 작업 후 매번 ===
git status
git add server/경로/파일.py
git commit -m "feat: 작업 내용 한 줄"
git push origin feature/본인브랜치

# === main 최신 반영 (merge 후) ===
git checkout main
git pull origin main
git checkout feature/본인브랜치
git merge main
```

**기억할 것 3가지:**  
① `main`에 직접 push 금지  
② `.env` 절대 commit 금지  
③ 작업은 **본인 `feature/` 브랜치**에서만

---

## 15. PR과 Gemini (코드 컨벤션 자동 리뷰)

PR을 올리면 AI가 `rules/CONVENTIONS.md` 기준으로 리뷰하게 하려면 **Gemini Code Assist** GitHub 앱을 씁니다.

### 15-1. 팀 리더 1회 설정

1. https://github.com/marketplace/gemini-code-assist → **Install**
2. 조직/계정 `junyeop10` 저장소 `-` 선택 → Install
3. Google 계정으로 로그인·권한 허용

### 15-2. 저장소에 이미 있는 파일 (커밋 필요)

| 파일 | 역할 |
|------|------|
| `.gemini/styleguide.md` | 리뷰 시 지킬 컨벤션 (CONVENTIONS 요약) |
| `.gemini/config.yaml` | 리뷰 on/off, 무시할 경로 (`.env`, `uploads/` 등) |

`main`에 merge 된 뒤부터 새 PR에 자동 적용됩니다.

### 15-3. PR에서 쓰는 법

| 하고 싶은 일 | 방법 |
|--------------|------|
| PR 열면 자동 리뷰 | 기본 (5분 내 코멘트) |
| 다시 리뷰 요청 | PR 댓글에 `/gemini review` |
| 요약만 | `/gemini summary` |
| 질문 | `@gemini-code-assist 이 부분 왜 그래요?` |

제안된 코드는 GitHub에서 **Commit suggestion** 으로 반영 가능 (초보자는 리뷰어랑 상의 후 적용).

### 15-4. PR 올리기 전 (로컬, 선택)

Cursor에서 Gemini 모델 선택 → `rules/CONVENTIONS.md` 첨부 후 “이 diff 컨벤션 맞는지 봐줘” (GitHub 없이 미리 검사).

### 15-5. 주의

- 무료 tier: PR 리뷰 **하루 약 33건** 제한 (팀 규모 참고)
- Gemini가 **코드를 직접 고쳐 merge 하지는 않음** — 코멘트·제안만 (최종은 사람이 merge)
- `.gemini/` 설정 변경도 PR로 올려야 반영됨
