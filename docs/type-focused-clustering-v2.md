# 문서 타입 중심 군집화 v2

## 목표

`input_files` 하위 폴더는 오프라인 검증 정답으로만 사용한다. 폴더명은 임베딩 입력이나 군집화 벡터에 넣지 않는다.

목표는 문서 근거를 바탕으로 유사한 파일을 먼저 군집화하고, 애매한 파일은 `Noise / API review`로 분리한 뒤, 이후 군집 대표 문서를 AI에 보내 카테고리 이름을 정할 수 있게 하는 것이다.

## 적용 흐름

```text
파일 읽기
→ 텍스트 부족 PDF만 OCR fallback, OCR cache 사용
→ evidence 생성
→ type_embedding_text 생성
→ paraphrase-multilingual-MiniLM-L12-v2 임베딩
→ pattern_vector 생성
→ text embedding 80% + pattern vector 20% 결합
→ L2 normalize
→ PCA 20차원 축소
→ HDBSCAN 세부 군집 생성
→ centroid cosine >= 0.98인 거의 동일한 군집만 병합
→ HDBSCAN이 noise로 판단한 문서는 Noise / API review로 이동
→ 세부 군집 centroid를 complete-link cosine >= 0.90 기준으로 상위 타입 후보 그룹화
→ 세부 군집 payload와 상위 타입 후보 payload 생성
```

## 모델과 선택 이유

임베딩 모델은 `paraphrase-multilingual-MiniLM-L12-v2`를 사용한다.

- 한국어와 영어가 섞인 문서명을 함께 처리할 수 있다.
- 대형 모델보다 가벼워 로컬 GUI에서 반복 실행하기 적합하다.
- 기존 프로젝트의 embedding cache와 호환된다.

`type_embedding_text`는 회사명, 주소, 연락처, 반복 푸터 같은 노이즈를 감쇠하고 문서 타입을 설명하는 제목, 파일명 타입 단어, 패턴 신호, 정제된 본문 일부를 앞쪽에 배치한다.

`pattern_vector`는 무거운 NLP 없이 기존 evidence와 regex를 재사용한다.

- 금액과 날짜 밀도
- 사업자등록번호
- 계약 조항과 법률 표현
- 영수증, 청구, 회의록, 결재, 신청서, 증명서, 발주서 신호

layout과 structure는 evidence에 계속 남긴다. 다만 모든 파일에서 안정적으로 얻을 수 없고, 현재 검증에서는 벡터에 강제 결합했을 때 더 좋아지지 않았기 때문에 필수 군집화 입력으로 사용하지 않는다.

## 군집화 설정

| 항목 | 값 |
|---|---|
| 기본 reducer | `PCA` |
| PCA `n_components` | 최대 `20` |
| HDBSCAN `metric` | `euclidean` |
| HDBSCAN `min_cluster_size` | `2` |
| HDBSCAN `min_samples` | `1` |
| HDBSCAN `cluster_selection_method` | `leaf` |
| centroid merge threshold | `0.98` |
| 상위 타입 후보 그룹화 | complete-link cosine `0.90` |
| Noise 이동 probability | 별도 강제 임계값 없음 |

PCA 결과를 L2 normalize한 뒤 HDBSCAN의 euclidean 거리로 군집화한다. 정규화된 벡터에서 euclidean 거리는 cosine 거리와 일관된 방향으로 동작한다. UMAP은 비교 실험 옵션으로 남겨 두되, 로컬 대화형 실행에서는 초기 준비 시간이 길어 기본값으로 사용하지 않는다.

## 실제 검증 결과

2026-06-01에 `input_files` 전체를 CLI로 실행했다.

```powershell
.\.venv\Scripts\python.exe app.py classify --input-dir .\input_files --output .\outputs\type_v2_final_filtered --evidence-workers 4
```

전체 실행 파일은 `88개`다. 하위 폴더명은 군집화 입력이 아니라 오프라인 참고 평가에만 사용했다.

| 지표 | 결과 |
|---|---:|
| 세부 군집 수 | `26개` |
| 파일 2개짜리 세부 군집 | `12개` |
| 상위 타입 후보 그룹 수 | `9개` |
| noise 문서 | `11개` |
| 배정 문서 기준 폴더 purity | `79.22%` |
| ARI | `0.1594` |

purity는 작은 군집이 많아도 높게 나올 수 있으므로 문서 타입 군집화 정확도와 동일하게 해석하면 안 된다. 현재 구조는 세부 군집을 보존하고, 유사한 세부 군집을 상위 타입 후보로 다시 묶은 뒤 AI가 대표 문서 근거를 보고 이름을 정하도록 설계한다.

최종 실행 산출물은 `outputs/type_v2_parent_groups`에 있다.

## 성능

성능 최적화 후에는 파일 hash 기반 evidence cache와 embedding cache를 함께 사용한다. 기존 파일만 다시 실행할 때 텍스트 추출을 반복하지 않는다. 새 파일은 thread worker 4개로 evidence를 추출한다.

| 50개 GUI형 실행 | 시간 |
|---|---:|
| 신규 파일형 실행, 모델 preload 및 파일 I/O warmup 완료 | `6.21초` |
| evidence + embedding cache 재실행 | `1.73초` |

GUI는 시작 직후 임베딩 모델을 백그라운드 preload한다. 로컬에 모델이 있으면 `local_files_only=True`로 먼저 로드하여 불필요한 네트워크 확인을 피한다. 최초 설치처럼 로컬 모델이 없을 때만 다운로드 fallback을 사용한다.

OneDrive 파일을 처음 읽는 직후에는 디스크 및 동기화 상태에 따라 신규 파일형 실행이 일시적으로 `11초` 안팎까지 늘어날 수 있다. 같은 GUI 세션에서 모델과 파일이 준비된 이후의 50개 목표는 `10초` 이내다.

## 주요 코드

- `src/categories_cluster_pipeline.py`: 활성 타입 중심 군집화
- `src/type_embedding_builder.py`: 노이즈 감쇠와 `type_embedding_text`
- `src/feature_vector_builder.py`: 가벼운 `pattern_vector`
- `src/evidence_pipeline.py`: 텍스트 추출, 선택적 구조 추출, OCR fallback
- `src/clustering_support.py`: 세부 군집 summary와 상위 타입 후보 그룹화
- `tools/evaluate_cluster_quality.py`: 폴더 정답을 오프라인 평가에만 사용하는 비교 도구

## 해석 시 주의사항

이 평가는 문서 타입 군집이 하위 폴더 구성과 얼마나 비슷한지 측정한 것이다. 폴더 하나 안에도 계약서, 견적서, 세금계산서처럼 실제 문서 타입이 여러 개 있을 수 있으므로, 세부 군집이 폴더 수보다 많아지는 것은 자연스럽다. 이후 AI는 대표 문서와 공통 근거를 보고 여러 세부 군집을 같은 상위 카테고리로 묶을 수 있다.
