# CHANGELOG

프로젝트 폴더에서 진행한 작업을 날짜 기준으로 기록하는 문서입니다.

## 기록 원칙

- 기능 추가, 구조 변경, 문서 정리, GitHub 반영 내용을 날짜 순서대로 기록합니다.
- 각 항목에는 가능한 한 `무엇을`, `왜`, `검증 여부`를 남깁니다.
- 세부 코드 변경은 Git diff와 커밋 기록으로 확인하고, 이 문서는 작업 로그와 발표 증빙용으로 함께 사용합니다.

## 2026-05-13

### 저장소 정리 및 GitHub 연결

- Gemini/외부 LLM 호출 관련 코드를 제거해 저장소에 민감한 호출 코드가 남지 않도록 정리함.
- 더 이상 쓰지 않는 보조 파일과 백업 파일을 삭제하고 저장소 기준 구조를 정리함.
- `README.md`를 현재 프로젝트 범위에 맞게 다시 정리함.
- `.gitignore`를 보강해 로컬 백업, scratch 성격 파일, 테스트 산출물이 GitHub에 올라가지 않도록 정리함.
- `tests.test_feedback_loop` 테스트 통과 확인.
- GitHub 원격 저장소와 연결해 초기 정리 내용을 반영함.

### 한글 룰 복구 및 정확도 개선

- `data/categories.json`의 한글 카테고리와 키워드를 정상화함.
- `src/rule_classifier.py`의 문맥 룰과 약한 키워드 보정 로직을 정리함.
- `src/text_cleaner.py`의 한글 토큰 처리 정규식을 복구함.
- `청구서`, `사업자등록증`, `법인등기부등본` 등 실제 문서 유형에 맞는 카테고리와 키워드를 보강함.
- `tests/test_rule_classifier.py`를 추가해 규칙 기반 분류를 검증함.

### 로컬 LLM 보조 분류 추가

- `src/llm_support.py`에 Ollama 기반 `qwen2.5:3b` 호출 로직을 추가함.
- `--use-llm`, `--llm-model` 옵션을 추가해 명시적으로 켰을 때만 LLM이 작동하게 함.
- confidence가 높은 문서는 LLM을 건너뛰고, 애매한 문서만 LLM으로 보조 판단하게 함.
- LLM 실패 시 기존 rule/embedding 결과를 유지하는 fallback 구조를 적용함.
- `tests/test_llm_support.py`를 추가하고 관련 테스트를 통과시킴.

## 2026-05-15

### OCR fallback 추가

- `rapidocr_onnxruntime`, `Pillow`를 프로젝트 의존성에 추가함.
- `src/ocr_support.py`를 새로 만들어 스캔 PDF OCR fallback 로직을 분리함.
- 일반 텍스트 추출 결과가 비어 있는 PDF만 OCR 대상으로 수집되도록 구현함.
- OCR은 최대 5페이지까지만 수행하고, 여러 실패 PDF를 `ProcessPoolExecutor`로 병렬 처리하도록 구현함.
- OCR로 텍스트를 얻은 뒤 규칙 점수를 다시 계산해 기존 분류 흐름에 반영함.
- 분류 출력과 근거 문자열에 OCR 사용 여부가 보이도록 표시를 추가함.
- `tests/test_ocr_fallback.py`를 추가하고 동작을 검증함.

### OCR fallback 최적화

- `RapidOCR()` 초기화를 전역 lazy singleton 구조로 바꿔 프로세스마다 1회만 초기화되도록 조정함.
- `should_run_ocr(...)`, `explain_ocr_decision(...)`로 OCR 실행 조건을 분리함.
- `사업자등록증`, `법인등기부등본` 계열은 파일명만으로도 강하게 분류되도록 evidence 기반 힌트를 추가함.
- `--ocr-workers`, `--ocr-min-chars` 옵션을 추가해 OCR 워커 수와 최소 텍스트 길이 기준을 조절할 수 있게 함.
- OCR 실행, 생략, 실패 여부를 로그에 남기도록 보강함.

### 파일명 기반 증빙 문서 힌트 확장

- `벤처기업인증서`, `지방세완납증명서`, `중소기업확인서`, `재무제표증명`까지 파일명 힌트를 확장함.
- 제목에 `인증서`, `증명서`, `확인서`, `등본`, `등록증` 등이 포함된 문서는 OCR 없이도 더 잘 분류되도록 강화함.
- 결과적으로 OCR 대상 수를 줄여 전체 처리 시간을 크게 단축함.

### GUI 상태 표시 개선

- OCR을 실제 사용한 파일은 GUI 결과표에서 회색으로 보이도록 표시를 추가함.
- GUI 시작 시 별도 로딩 화면 대신 메인 화면을 먼저 띄우고, 임베딩 모델은 백그라운드에서 로드되도록 조정함.
- 상태 표시를 통해 임베딩 로딩 진행 여부를 사용자가 확인할 수 있게 함.

### 입력 형식 확장 및 영어 키워드 보강

- `python-docx`, `openpyxl`, `python-pptx`를 추가해 `docx`, `xlsx`, `pptx` 읽기를 지원함.
- `src/office_reader.py`를 추가하고 `src/file_reader.py`와 연결함.
- 파일 탐색 및 GUI 드래그 앤 드롭 대상도 `docx`, `xlsx`, `pptx`까지 확장함.
- `data/categories.json`에 영어 키워드를 추가해 일부 영문 문서도 규칙 기반으로 분류 가능하게 보강함.
- `tests/test_file_reader.py`를 추가해 오피스 문서 추출과 파일 탐색 동작을 검증함.

### 버전 기록 문서 체계 정리

- 발표와 작업 증빙을 위해 `docs/version-history.md`를 버전 단위 요약 문서로 정리함.
- `CHANGELOG.md`는 날짜 중심 상세 작업 로그로 유지하고, `docs/version-history.md`는 발표용 요약 문서로 역할을 분리함.

### 파일별 읽기 방식과 모델 기록 문서화

- 발표 자료와 작업 증빙에 바로 활용할 수 있도록 파일 형식별 추출 방식, 사용 라이브러리, 사용 모델을 문서에 명시함.
- `txt`, `pdf`, `docx`, `xlsx`, `pptx`가 각각 어떤 방식으로 읽히는지 정리함.
- OCR fallback의 실행 조건, 최대 5페이지 제한, RapidOCR 재사용 구조를 기록함.
- 임베딩 모델 `paraphrase-multilingual-MiniLM-L12-v2`, 로컬 LLM `qwen2.5:3b`, PDF/오피스 추출 라이브러리도 버전 문서에 명시함.
