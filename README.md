# 파일 분류 MVP

사용자 피드백을 바탕으로 문서 분류 정확도를 점진적으로 개선하는 Python 기반 문서 분류 MVP입니다.

현재 목표는 모델 fine-tuning 없이 아래 흐름을 안정적으로 운영하는 것입니다.

```text
파일 읽기 -> evidence_text 추출 -> 룰/문맥 기반 1차 분류 -> 사용자 검토/수정 -> SQLite 저장 -> 다음 분류에 반영
```

## 현재 범위

- 지원 입력: `txt`, `pdf`
- 실행 방식: CLI, Tkinter GUI
- 분류 방식: 룰 기반 + 피드백 기반 보정 + 선택적 임베딩 유사도
- 저장 방식: SQLite
- 중복 처리: `xxhash64` 기반 중복 감지
- OCR: 아직 미구현

## 주요 기능

- SQLite 기반 분류 이력 저장
- `files`, `classifications`, `feedback_logs`, `confirmed_examples`, `rules` 테이블 생성
- txt/pdf 텍스트 추출
- 긴 문서에서 처음/중간/마지막 구간을 샘플링해 최대 4500자 evidence text 생성
- 키워드 룰 + 문맥 룰 기반 1차 분류
- 사용자 확정/수정 결과 저장
- 반복 수정 패턴 기반 룰 후보 제안
- `--fast` 모드에서 멀티프로세싱 기반 빠른 분류
- Tkinter GUI 및 드래그 앤 드롭 지원

## 현재 분류 로직

- 강한 문맥 룰이 있으면 우선 추천합니다.
- 약한 단일 키워드만 맞는 경우 `검토필요`로 보냅니다.
- `--fast` 모드에서는 속도를 우선해 임베딩을 대부분 생략합니다.
- 일반 모드에서는 확정 예시가 쌓인 뒤 sentence-transformers 기반 유사도 보정을 사용할 수 있습니다.

예시 문맥 룰:

- `모집 + 신청 + 접수` -> 공고/공고문
- `공고 + 지원 + 제출서류` -> 공고/공고문
- `갑 + 을 + 계약기간` -> 계약서
- `과업내용 + 용역목적` -> 과업지시서
- `세금계산서 + 공급가액 + 합계금액` -> 청구서

## 프로젝트 구조

```text
app.py                  CLI 시작점
app_gui.py              GUI 시작점
run_gui.bat             GUI 실행 배치 파일
requirements.txt        설치 패키지 목록
data/categories.json    기본 카테고리/키워드
src/cli.py              CLI 명령 처리
src/classifier.py       최종 점수 계산과 추천
src/rule_classifier.py  키워드/문맥 룰 점수 계산
src/fast_worker.py      fast 모드 worker 처리
src/vectorizer.py       임베딩 생성과 코사인 유사도 계산
src/storage.py          SQLite 저장소
src/file_reader.py      txt/pdf 읽기 진입점
src/pdf_reader.py       PDF 텍스트 추출
src/text_cleaner.py     텍스트 정리와 샘플링
src/hash_utils.py       xxhash 계산
src/feedback.py         피드백 기반 룰 후보 분석
src/gui.py              Tkinter GUI
tests/                  피드백 루프 테스트
```

## 설치

Python 3.11 이상을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 실행

DB 초기화:

```powershell
.\.venv\Scripts\python.exe app.py init-db
```

빠른 분류:

```powershell
.\.venv\Scripts\python.exe app.py classify --fast
```

검토하면서 분류:

```powershell
.\.venv\Scripts\python.exe app.py classify --fast --review
```

일반 분류:

```powershell
.\.venv\Scripts\python.exe app.py classify --review
```

룰 후보 제안:

```powershell
.\.venv\Scripts\python.exe app.py suggest-rules
```

DB 통계:

```powershell
.\.venv\Scripts\python.exe app.py stats
```

GUI 실행:

```powershell
.\run_gui.bat
```

## 입력 파일

`input_files` 폴더에 `txt`, `pdf` 파일을 넣고 실행합니다.

실제 업무 문서와 개인 문서는 GitHub에 올리지 않도록 `.gitignore`에서 제외합니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_feedback_loop -v
```

검증 내용:

- 사용자가 추천 결과를 수정/확정하면 `feedback_logs`에 저장되는지 확인
- 확정 결과가 `confirmed_examples`에 저장되는지 확인
- 이후 분류에서 feedback/embedding 점수가 반영되는지 확인

## GitHub 업로드 가이드

이 저장소에는 코드와 설정만 올리고, 실제 문서와 로컬 실행 산출물은 제외하는 방식을 권장합니다.

현재 `.gitignore`로 제외되는 항목:

- `.venv/`
- `data/*.db`
- `input_files/` 내부 실제 문서
- `tests_runtime/`
- `scratch/`
- `__pycache__/`
- 로컬 백업 폴더와 원본 복사본 파일

권장 업로드 대상:

- `src/`
- `tests/`
- `data/categories.json`
- `README.md`
- `requirements.txt`
- 실행 진입점 파일

기본 업로드 순서:

```powershell
git status
git add .
git commit -m "Clean up LLM placeholder and update README"
git push
```

## 현재 제한

- OCR은 아직 구현하지 않았습니다.
- 모델 fine-tuning은 하지 않습니다.
- 현재 입력 포맷은 `txt`, `pdf` 중심입니다.
