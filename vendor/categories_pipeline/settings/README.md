# 의미기반 분류 — 기초 코드

설계 문서(`의미기반 분류 .hwpx`)의 4단계 + 운영 시스템 구조.
클래스 기반 모듈로 신규 파일 점진 처리(approximate predict), 모델 저장/로드,
피드백 DB 누적까지 지원한다.

## 파이프라인 순서

| 단계 | 파일 | 클래스 | 역할 |
| --- | --- | --- | --- |
| 1 | `embedder.py` | `Embedder` | 임베딩 (다국어 384-dim, 3구간 가중 평균 0.5/0.25/0.25) |
| 2 | `reducer.py` | `DimReducer` | UMAP 차원 축소 (기본 15차원, cosine metric) |
| 3 | `clusterer.py` | `Clusterer` | HDBSCAN — 프로젝트 단위 군집화 |
| 4 | `similarity.py` | `SimilarityClassifier` | cosine similarity vs 피드백 DB (threshold 0.75) |

부속 파일:
- `categories.json` — 분류 카테고리 정의 (가변 N개)
- `evidence.py` — 문서별/단계별 `EvidencePackage` (신뢰도·근거·학습·LLM 위임)
- `timing.py` — 단계별 실행 시간 측정 (`StepTimer`)
- `main.py` — 1→2→3→4 순서로 호출하는 오케스트레이터

## 핵심 특징

- **다국어 임베딩**: `paraphrase-multilingual-MiniLM-L12-v2` (한국어 포함 50+ 언어)
- **3구간 가중 평균**: 문서 front/middle/rear을 0.5/0.25/0.25로 통합 → 앞부분 핵심정보 강조
- **신규 파일 점진 처리**: `DimReducer.transform`, `Clusterer.predict` (approximate_predict)
- **피드백 누적 학습**: `EmbeddingStore.add()` 1건 = 즉시 반영 (재학습 불필요)
- **3-stage cascade의 2단계**: 신뢰도 미달 시 LLM 단계로 위임
- **카테고리 수 가변**: `categories.json`에서 N개 정의

## 설치

```powershell
pip install -r requirements.txt
```

> Windows에서 `hdbscan`/`umap-learn` 빌드 실패 시:
> `pip install --upgrade pip setuptools wheel` 후 재시도.

## 실행

```powershell
# 내장 데모 데이터로 동작 확인
python main.py --demo

# 문서별 분류 근거(EvidencePackage.explain) 출력
python main.py --demo --explain

# EvidencePackage를 JSONL로 저장
python main.py --demo --dump evidence_out.jsonl

# 단계별 시간 측정 JSON으로 저장
python main.py --demo --timing-json timing.json

# 실제 입력 (3구간 분리)
python main.py --input docs.jsonl

# 다른 카테고리 체계 / 임계값
python main.py --input docs.jsonl --categories my_categories.json --threshold 0.70
```

## EvidencePackage (문서별 증거 패키지)

문서 1건마다 모든 단계 산출물을 모은 객체. 네 가지 용도:

| 메서드/속성 | 용도 |
| --- | --- |
| `pkg.overall_confidence` | similarity 점수 × cluster 확신도 보정 → 통합 신뢰도 (noise면 30% 페널티) |
| `pkg.explain()` | 사람이 읽을 분류 근거 텍스트 (감사/사용자 확인용) |
| `pkg.to_training_record(category)` | 사용자 확정 라벨 + 벡터 → `EmbeddingStore.add()`용 페이로드 |
| `pkg.to_llm_payload()` | 신뢰도 미달 시 LLM에 넘길 컨텍스트 (원문 + top-k + 사유) |
| `pkg.to_json()` | 디스크 적재용 직렬화 |

## 입력 형식

권장 — 3구간 분리(EvidencePackage 형태):
```jsonl
{"doc_id": "사업계획서.pdf", "front": "전반 텍스트", "middle": "중반", "rear": "후반"}
```

또는 단일 텍스트(자동 1:1:1 분할):
```jsonl
{"doc_id": "공고문.hwpx", "text": "전체 본문..."}
```

## 카테고리 정의 (가변 N개)

분류 폴더 수는 **고정이 아니다**. `categories.json`을 수정하면 N개 어떤 수든 동작한다.

```json
[
  {"folder": "1. 공고_지침_양식", "description": "사업 공고문, 모집 안내, ..."},
  {"folder": "2. 사업계획서 수행계획서", "description": "..."}
]
```

- `folder`: 분류 결과 폴더명 (그대로 출력됨)
- `description`: cold-start 시드용 자연어 설명 — 운영 단계에서 사용자 확정 이력이
  쌓이면 이 설명 기반 시드보다 실제 파일 벡터가 우선시된다.

기본값 `categories.json`은 설계 문서의 "분류 예시" 8개 폴더로 채워져 있다.

## 운영 시 흐름

```
신규 파일 도착
   ↓
Embedder.embed_evidence(front, middle, rear)   ← 384차원 벡터
   ↓
DimReducer.transform(vector)                   ← 15차원으로 축소
   ↓
Clusterer.predict(reduced)                     ← 프로젝트 군집 ID (-1=noise)
   ↓
SimilarityClassifier.classify(vector)          ← 카테고리 + 신뢰도
   ↓
신뢰도 ≥ 0.75 → 분류 확정 + EmbeddingStore.add(category, vector)
신뢰도 < 0.75 → LLM 단계로 위임
```

## 파라미터 튜닝 메모

- **UMAP 차원** (`reducer.py:N_COMPONENTS_MAX`): 기본 15. 카테고리가 의미적으로
  가까워 더 잘게 나누고 싶으면 20~25, 거꾸로 잡음이 많으면 10.
- **HDBSCAN `min_cluster_size`** (`clusterer.py:MIN_CLUSTER_SIZE`): 기본 3.
  문서가 수천 건 이상이면 5~10.
- **cosine 임계값** (`similarity.py:CONFIDENCE_THRESHOLD`): 기본 0.75.
  운영 초기 데이터가 적을 때는 0.70으로 살짝 낮춰도 됨.
