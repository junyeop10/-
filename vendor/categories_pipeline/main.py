"""의미기반 분류 파이프라인 — 클래스 기반 오케스트레이터.

설계 문서(`의미기반 분류 .hwpx`)의 4단계 순서:
  STEP 1. 임베딩 (Embedder, 다국어 384-dim, 3구간 가중 평균)
  STEP 2. 차원 축소 (DimReducer / UMAP, 기본 15차원)
  STEP 3. HDBSCAN (Clusterer, 프로젝트 단위 grouping)
  STEP 4. cosine similarity (SimilarityClassifier, 피드백 DB 기반)

문서마다 EvidencePackage 생성, 단계별 시간 측정, 군집·미분류 사유 설명.
--input-dir + --output-dir 모드: 실제 폴더 구조로 파일 복사.

사용:
  python main.py --demo
  python main.py --input-dir 경로 --output-dir 경로
  python main.py --input-dir 경로 --output-dir 경로 --explain --report out.md
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Windows 콘솔(cp949)에서도 한글/유니코드 출력이 깨지지 않도록 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from cluster_explainer import explain_clusters, explain_unclassified
from clusterer import Clusterer
from embedder import EMBED_DIM, EmbedResult, Embedder
from evidence import (
    ClusterEvidence,
    EmbeddingEvidence,
    EvidencePackage,
    ReduceEvidence,
    SimilarityEvidence,
    build_cluster_evidence,
    build_embedding_evidence,
    build_evidence_package,
    build_reduce_evidence,
    build_similarity_evidence,
)
from file_reader import read_file, split_three
from reducer import DimReducer
from rule_classifier import RuleResult, classify_by_filename
from semantic_core import (
    DEFAULT_KNN_K,
    DEFAULT_THRESHOLD as SEMANTIC_DEFAULT_THRESHOLD,
    SemanticClassifier,
    SemanticStore,
)
from similarity import (
    Category,
    EmbeddingStore,
    SimilarityClassifier,
    load_categories,
)
from timing import StepTimer
from embed_cache import EmbeddingCache


SUPPORTED_EXTS = {".pdf", ".hwp", ".hwpx", ".docx", ".xlsx", ".txt", ".md"}


@dataclass
class InputDoc:
    doc_id: str            # 표시용 (보통 파일명)
    source_path: Path | None  # 실제 파일 경로 (디렉터리 모드일 때)
    front: str
    middle: str
    rear: str
    read_method: str = "manual"

    @property
    def raw_text(self) -> str:
        # split_three는 연속 슬라이스이므로 구분자 없이 이어붙이면 원문 그대로 복원.
        # (\n 구분자를 넣으면 분할 경계에서 "사업자등록증"→"사업자등"+"록증"처럼
        #  키워드가 쪼개져 BM25 어휘 매칭이 깨진다 — 짧은 파일명-fallback 문서에 치명적)
        return self.front + self.middle + self.rear


# ──────────────────────────────────────────────
# 입력 로더
# ──────────────────────────────────────────────

def load_jsonl(path: Path) -> list[InputDoc]:
    docs: list[InputDoc] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "front" in obj and "middle" in obj and "rear" in obj:
                docs.append(InputDoc(
                    doc_id=str(obj["doc_id"]), source_path=None,
                    front=str(obj.get("front", "")),
                    middle=str(obj.get("middle", "")),
                    rear=str(obj.get("rear", "")),
                ))
            else:
                f0, m0, r0 = split_three(str(obj.get("text", "")))
                docs.append(InputDoc(
                    doc_id=str(obj["doc_id"]), source_path=None,
                    front=f0, middle=m0, rear=r0,
                ))
    return docs


def load_directory(dirpath: Path) -> list[InputDoc]:
    """디렉터리 안의 모든 지원 포맷 파일을 읽어 InputDoc 리스트로."""
    docs: list[InputDoc] = []
    for p in sorted(dirpath.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            # 미지원 확장자도 파일명만으로 처리
            pass
        r = read_file(p)
        f0, m0, r0 = split_three(r.text)
        docs.append(InputDoc(
            doc_id=p.name,
            source_path=p,
            front=f0, middle=m0, rear=r0,
            read_method=r.source_method,
        ))
    return docs


def demo_documents() -> list[InputDoc]:
    raw = [
        ("a_공고.txt", "2026년 청년 창업 지원사업 공고문입니다. 신청 자격 및 제출 양식 안내."),
        ("b_지침.txt", "사업 운영 지침서. 제출 서식과 가이드라인을 정리한 안내문."),
        ("c_사업계획서.txt", "스타트업 사업계획서 — 사업 개요, 추진 일정, 예산 계획 포함."),
        ("d_수행계획.txt", "연구 과제 수행계획서. 단계별 추진 일정과 인력 운영 계획."),
        ("e_조사.txt", "국내외 시장 조사 보고. 산업 동향과 벤치마킹 사례 정리."),
        ("f_보고서.txt", "프로젝트 최종 보고서 — 성과 정리, 분석 결과, 결론."),
        ("g_발표.txt", "킥오프 발표자료 슬라이드. PT 데모 자료 및 브리핑."),
        ("h_견적.txt", "용역 견적서 및 세금계산서. 정산 명세와 비용 청구 내역."),
        ("i_인증.txt", "벤처기업 확인서와 사업자등록증 사본 — 자격 증명."),
        ("j_기타.txt", "개인 메모. 회의 잡담 기록."),
    ]
    docs = []
    for did, t in raw:
        f0, m0, r0 = split_three(t)
        docs.append(InputDoc(
            doc_id=did, source_path=None,
            front=f0, middle=m0, rear=r0,
        ))
    return docs


# ──────────────────────────────────────────────
# Cold-start seed
# ──────────────────────────────────────────────

def seed_feedback_store(
    store: EmbeddingStore,
    categories: list[Category],
    embedder: Embedder,
) -> None:
    """Cold-start seed: 카테고리 description을 임베딩해 1건씩 주입."""
    descriptions = [f"{c.folder}: {c.description}" for c in categories]
    results = embedder.embed_batch(descriptions)
    for cat, r in zip(categories, results):
        store.add(cat.folder, r.vector)
    logging.info("seed feedback: %d개 카테고리 cold-start 시드 주입", len(categories))


def seed_from_groundtruth(
    store: EmbeddingStore,
    embedder: Embedder,
    seed_dir: Path,
    per_category: int,
) -> set[str]:
    """정답지 디렉터리에서 카테고리별 per_category개를 시드로 EmbeddingStore에 주입.

    같은 파일이 분류 대상으로 다시 들어가지 않도록 시드로 쓴 파일명 set을 반환.
    """
    seeded: set[str] = set()
    n_seeds = 0
    for cat_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in cat_dir.iterdir() if f.is_file())
        picked = files[:per_category]
        for f in picked:
            try:
                r = read_file(f)
            except Exception as e:  # noqa: BLE001
                logging.warning("seed 읽기 실패 [%s] %s: %s", cat_dir.name, f.name, e)
                continue
            front, middle, rear = split_three(r.text)
            ev = embedder.embed_evidence(front, middle, rear)
            store.add(cat_dir.name, ev.vector)
            seeded.add(f.name)
            n_seeds += 1
        logging.info("seed [%s] %d건 주입 (시드 파일은 분류에서 제외)",
                     cat_dir.name, len(picked))
    logging.info("정답지 시드 총 %d건 주입 완료", n_seeds)
    return seeded


# ──────────────────────────────────────────────
# [우선순위 3] 배치 임베딩 + 캐시
# ──────────────────────────────────────────────

def embed_docs_weighted(
    embedder: Embedder,
    docs: list[InputDoc],
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
    cache: "EmbeddingCache | None" = None,
):
    """문서 N건의 3구간(front/middle/rear)을 *한 번의 배치 encode*로 처리.

    per-doc embed_evidence 루프(3N개를 N번 나눠 encode)보다 모델 호출
    오버헤드가 적어 빠르다. 캐시가 있으면 본문 해시로 적중분을 건너뛴다.
    결과는 embed_evidence와 수치적으로 동일.
    """
    if not docs:
        return []
    n = len(docs)
    vectors: list[np.ndarray | None] = [None] * n

    # 1) 캐시 조회
    keys = [cache.key(d.front, d.middle, d.rear) if cache else None for d in docs]
    miss_idx: list[int] = []
    for i, d in enumerate(docs):
        if cache is not None:
            hit = cache.get(keys[i])
            if hit is not None:
                vectors[i] = hit
                continue
        miss_idx.append(i)

    # 2) 미스만 배치 임베딩 (3구간을 한 배치로)
    if miss_idx:
        texts: list[str] = []
        for i in miss_idx:
            texts.extend([docs[i].front, docs[i].middle, docs[i].rear])
        flat = embedder.embed_batch(texts)                      # 3·len(miss)
        mat = np.stack([r.vector for r in flat], axis=0).reshape(len(miss_idx), 3, EMBED_DIM)
        w = np.asarray(weights, dtype=np.float32)
        w = w / w.sum()
        combined = (mat * w[None, :, None]).sum(axis=1)         # (m, D)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        combined = (combined / norms).astype(np.float32)
        for j, i in enumerate(miss_idx):
            vectors[i] = combined[j]
            if cache is not None:
                cache.put(keys[i], combined[j])

    return [
        EmbedResult(vector=vectors[i], text_preview=docs[i].front[:80])
        for i in range(n)
    ]


# ──────────────────────────────────────────────
# [우선순위 1] 의미 코어용 시드 (벡터 + 본문 텍스트)
# ──────────────────────────────────────────────

def seed_semantic_descriptions(
    store: "SemanticStore",
    categories: list[Category],
    embedder: Embedder,
) -> None:
    """카테고리 description을 (벡터, 텍스트)로 cold-start 시드 주입."""
    descs = [f"{c.folder}: {c.description}" for c in categories]
    results = embedder.embed_batch(descs)
    for cat, r, txt in zip(categories, results, descs):
        store.add(cat.folder, r.vector, text=txt)
    logging.info("의미 시드(desc): %d개 카테고리 주입", len(categories))


def seed_semantic_groundtruth(
    store: "SemanticStore",
    embedder: Embedder,
    seed_dir: Path,
    per_category: int,
    cache: "EmbeddingCache | None" = None,
) -> set[str]:
    """정답지에서 카테고리별 per_category개를 (벡터, 본문)으로 주입. BM25용 본문 저장."""
    seeded: set[str] = set()
    items: list[tuple[str, str, str]] = []   # (category, filename, text)
    for cat_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in cat_dir.iterdir() if f.is_file())
        for f in files[:per_category]:
            try:
                r = read_file(f)
            except Exception as e:  # noqa: BLE001
                logging.warning("seed 읽기 실패 [%s] %s: %s", cat_dir.name, f.name, e)
                continue
            items.append((cat_dir.name, f.name, r.text))
            seeded.add(f.name)
    if items:
        tmp = []
        for _cat, fname, text in items:
            f0, m0, r0 = split_three(text)
            tmp.append(InputDoc(doc_id=fname, source_path=None, front=f0, middle=m0, rear=r0))
        embs = embed_docs_weighted(embedder, tmp, cache=cache)
        for (cat, _fname, text), er in zip(items, embs):
            store.add(cat, er.vector, text=text)
    logging.info("의미 시드(정답지) 총 %d건 주입", len(items))
    return seeded


# ──────────────────────────────────────────────
# 실제 폴더 분류
# ──────────────────────────────────────────────

UNCLASSIFIED_FOLDER = "_미분류_LLM위임"


def _make_rule_only_package(doc: "InputDoc", rule_cat: str, rule_conf: float,
                            matched: list[str]) -> EvidencePackage:
    """룰 매칭 문서용 EvidencePackage. 임베딩은 건너뛰고 zero vector로 채움."""
    zero_vec = np.zeros(EMBED_DIM, dtype=np.float32)
    embed_ev = EmbeddingEvidence(
        model_name="(rule-matched, embedding skipped)",
        dim=EMBED_DIM, norm=0.0,
        text_preview=doc.doc_id[:80],
        front_chars=len(doc.front),
        middle_chars=len(doc.middle),
        rear_chars=len(doc.rear),
        weights=(0.5, 0.25, 0.25),
        vector=zero_vec,
    )
    reduce_ev = ReduceEvidence(input_dim=EMBED_DIM, output_dim=0, metric="(skipped)")
    cluster_ev = ClusterEvidence(label=-1, probability=1.0, cluster_size=0,
                                 is_noise=False)
    rule_label = f"rule:{','.join(matched[:3])}" if matched else "rule"
    sim_ev = SimilarityEvidence(
        decision_category=rule_cat,
        confidence=rule_conf,
        threshold=0.85,
        is_confident=True,
        top_k_categories=[(rule_cat, rule_conf), (rule_label, 0.0)],
        nearest_neighbors=[],
    )
    return EvidencePackage(
        doc_id=doc.doc_id, raw_text=doc.raw_text,
        embedding=embed_ev, reduce=reduce_ev,
        cluster=cluster_ev, similarity=sim_ev,
    )


def copy_to_classified_folders(
    output_root: Path,
    categories: list[Category],
    packages: list[EvidencePackage],
    input_docs: list[InputDoc],
) -> dict[str, list[str]]:
    """카테고리별 폴더 생성 + 실제 파일 복사. 반환: {folder: [doc_id, ...]}"""
    output_root.mkdir(parents=True, exist_ok=True)

    # 카테고리 폴더 미리 생성
    for c in categories:
        (output_root / c.folder).mkdir(exist_ok=True)
    (output_root / UNCLASSIFIED_FOLDER).mkdir(exist_ok=True)

    layout: dict[str, list[str]] = {c.folder: [] for c in categories}
    layout[UNCLASSIFIED_FOLDER] = []

    for pkg, d in zip(packages, input_docs):
        target = pkg.final_category if not pkg.needs_llm else UNCLASSIFIED_FOLDER
        layout.setdefault(target, []).append(pkg.doc_id)

        if d.source_path is None or not d.source_path.exists():
            continue
        dst = output_root / target / d.source_path.name
        try:
            shutil.copy2(d.source_path, dst)
        except Exception as e:  # noqa: BLE001
            logging.warning("복사 실패 %s → %s: %s", d.source_path, dst, e)

    return layout


# ──────────────────────────────────────────────
# 파이프라인 실행
# ──────────────────────────────────────────────

def run(
    docs: list[InputDoc],
    categories: list[Category],
    threshold: float,
    explain: bool,
    dump_path: Path | None,
    timing_json_path: Path | None,
    output_root: Path | None,
    report_path: Path | None,
    seed_dir: Path | None = None,
    seed_per_category: int = 5,
    use_description_seed: bool = True,
    enable_clustering: bool = False,
    cascade_low_threshold: float | None = None,
    use_rule_stage: bool = True,
    conservative_rules: bool = False,
    use_semantic_core: bool = True,
    semantic_threshold: float | None = None,
    knn_k: int = DEFAULT_KNN_K,
    cache_path: Path | None = None,
    use_cache: bool = True,
    online_learning: bool = False,
) -> list[EvidencePackage]:
    """파이프라인 cascade:
       STAGE A 룰 기반 (파일명 키워드)
       STAGE B 의미기반 코어 — 임베딩 → [옵션 UMAP+HDBSCAN] →
               [우선순위1] 4신호 융합(cos_max·centroid·kNN·BM25) + threshold cascade
       STAGE C LLM 위임 (둘 다 못 잡은 것)

    use_semantic_core=True(기본): STEP4에서 SemanticClassifier(4신호 융합) 사용.
    [우선순위3] 배치 임베딩 + 디스크 캐시(use_cache) + 시드 기본 5건.
    """
    if not docs:
        print("[!] 입력 문서가 없습니다.")
        return []

    timer = StepTimer()

    # ── 정답지 시드 제외 ─────────────
    if seed_dir is not None:
        seeded_names: set[str] = set()
        for cat_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            files = sorted(f for f in cat_dir.iterdir() if f.is_file())
            for f in files[:seed_per_category]:
                seeded_names.add(f.name)
        before = len(docs)
        docs = [d for d in docs if d.doc_id not in seeded_names]
        print(f"[SEED] 정답지 시드 {before - len(docs)}건 제외")

    n_input = len(docs)

    # ──────────────────────────────────────────────
    # STAGE A: 룰 기반 1차 분류 (파일명 키워드)
    # ──────────────────────────────────────────────
    rule_packages: list[EvidencePackage] = []
    semantic_docs: list[InputDoc] = []
    if use_rule_stage:
        print(f"[STAGE A] 룰 기반 1차 분류 — {n_input}개 "
              f"({'보수적 룰' if conservative_rules else '전체 룰'})")
        with timer.measure("STAGE A: 룰 기반"):
            for d in docs:
                r = classify_by_filename(d.doc_id, conservative=conservative_rules)
                if r.is_confident:
                    rule_packages.append(_make_rule_only_package(
                        d, r.category, r.confidence, r.matched_keywords))
                else:
                    semantic_docs.append(d)
        print(f"          룰 확정 {len(rule_packages)}건 / "
              f"의미기반 위임 {len(semantic_docs)}건")
    else:
        semantic_docs = list(docs)

    # ──────────────────────────────────────────────
    # STAGE B: 의미기반 (semantic_docs만)
    # ──────────────────────────────────────────────
    semantic_packages: list[EvidencePackage] = []
    cluster_result = None
    reduce_result = None
    doc_matrix = None
    embedder = None
    emb_cache: EmbeddingCache | None = None

    if semantic_docs:
        print(f"[STAGE B] 의미기반 분류 — {len(semantic_docs)}개")

        # STEP 1: 임베딩 (semantic_docs만) — [우선순위3] 배치 encode + 디스크 캐시
        with timer.measure("STAGE B / STEP 1: 임베딩"):
            with timer.measure("  embedder 로드", sub=True):
                embedder = Embedder()
            if use_cache:
                cpath = (cache_path or
                         (output_root / "_emb_cache.pkl" if output_root
                          else Path("_emb_cache.pkl")))
                emb_cache = EmbeddingCache(cpath, embedder.model_name, (0.5, 0.25, 0.25))
            with timer.measure("  embed_batch (전체)", sub=True):
                embed_results = embed_docs_weighted(
                    embedder, semantic_docs, cache=emb_cache)
            doc_matrix = np.stack([r.vector for r in embed_results], axis=0)
            if emb_cache is not None:
                print(f"           [캐시] {emb_cache.stats()}")

        # STEP 2+3: UMAP + HDBSCAN (옵션, 보조 신호용)
        if enable_clustering and len(semantic_docs) >= 5:
            print("           STEP 2~3: UMAP + HDBSCAN (보조 군집화)")
            with timer.measure("STAGE B / STEP 2: UMAP"):
                reducer = DimReducer()
                reduce_result = reducer.fit_transform(doc_matrix)
            with timer.measure("STAGE B / STEP 3: HDBSCAN"):
                clusterer = Clusterer()
                cluster_result = clusterer.fit_predict(reduce_result.vectors)
            print(f"           {cluster_result.n_clusters}개 군집, "
                  f"{cluster_result.n_noise}건 noise")

        # STEP 4: [우선순위1] 4신호 융합 의미 코어 + threshold cascade
        if use_semantic_core:
            base_thr = (semantic_threshold if semantic_threshold is not None
                        else SEMANTIC_DEFAULT_THRESHOLD)
        else:
            base_thr = threshold
        thr_low = (cascade_low_threshold if cascade_low_threshold is not None
                   else base_thr * (0.8 if use_semantic_core else 0.75))
        mode = ("융합(cos_max·centroid·kNN·BM25)" if use_semantic_core
                else "cosine-max 단일")
        print(f"           STEP 4: {mode} — primary {base_thr:.2f}, cascade {thr_low:.2f}")
        with timer.measure("STAGE B / STEP 4: 의미 융합 + cascade"):
            with timer.measure("  seed 주입", sub=True):
                if use_semantic_core:
                    store = SemanticStore()
                    if use_description_seed:
                        seed_semantic_descriptions(store, categories, embedder)
                    if seed_dir is not None:
                        seed_semantic_groundtruth(store, embedder, seed_dir,
                                                  seed_per_category, cache=emb_cache)
                else:
                    store = EmbeddingStore()
                    if use_description_seed:
                        seed_feedback_store(store, categories, embedder)
                    if seed_dir is not None:
                        seed_from_groundtruth(store, embedder, seed_dir, seed_per_category)
            if use_semantic_core:
                primary = SemanticClassifier(store, threshold=base_thr, knn_k=knn_k)
                cascade = SemanticClassifier(store, threshold=thr_low, knn_k=knn_k)
            else:
                primary = SimilarityClassifier(store, threshold=base_thr)
                cascade = SimilarityClassifier(store, threshold=thr_low)

            n_primary = n_cascade = n_llm = 0
            for i, (d, er) in enumerate(zip(semantic_docs, embed_results)):
                with timer.measure("  classify+cascade (per doc)", sub=True):
                    if use_semantic_core:
                        result = primary.classify(er.vector, d.raw_text)
                        if result.category is None:        # 1차 미달 → 캐스케이드
                            result = cascade.classify(er.vector, d.raw_text)
                    else:
                        result = primary.classify(er.vector)
                        if result.category is None:
                            result = cascade.classify(er.vector)
                    if result.category is not None:
                        if result.threshold >= base_thr:
                            n_primary += 1
                        else:
                            n_cascade += 1
                    else:
                        n_llm += 1
                    neighbors = primary.top_n_similar(er.vector, n=5)

                pkg = build_evidence_package(
                    doc_id=d.doc_id, raw_text=d.raw_text,
                    embedding=build_embedding_evidence(
                        er, d.front, d.middle, d.rear),
                    reduce=(build_reduce_evidence(reduce_result)
                            if reduce_result else
                            ReduceEvidence(input_dim=EMBED_DIM, output_dim=0,
                                          metric="(skipped)")),
                    cluster=(build_cluster_evidence(i, cluster_result)
                             if cluster_result else
                             ClusterEvidence(label=-1, probability=0.0,
                                            cluster_size=0, is_noise=False)),
                    similarity=build_similarity_evidence(result, neighbors),
                )
                semantic_packages.append(pkg)

                # 온라인 학습: 기본 OFF. 자동 예측 라벨을 store에 넣으면 cold-start
                # batch에서 오분류가 전파돼(자석 증폭) 정확도가 크게 떨어진다.
                # production에선 *사용자 확정* 라벨만 따로 add 하면 된다.
                if online_learning and not pkg.needs_llm:
                    rec = pkg.to_training_record(
                        confirmed_category=pkg.final_category, source="auto")
                    if use_semantic_core:
                        store.add(rec["category"], rec["vector"], text=d.raw_text)
                    else:
                        store.add(rec["category"], rec["vector"])
            print(f"           1차 확정 {n_primary} / 캐스케이드 확정 {n_cascade} "
                  f"/ LLM 위임 {n_llm}")

    # ──────────────────────────────────────────────
    # 통합: 입력 순서대로 packages 재정렬
    # ──────────────────────────────────────────────
    by_id = {p.doc_id: p for p in (rule_packages + semantic_packages)}
    packages = [by_id[d.doc_id] for d in docs if d.doc_id in by_id]

    # ── 군집/미분류 해설 (의미기반 docs만 대상) ───────
    cluster_explanations = []
    if cluster_result is not None and embedder is not None:
        cat_embs_full = embedder.embed_batch(
            [f"{c.folder}: {c.description}" for c in categories])
        cat_emb_matrix = np.stack([r.vector for r in cat_embs_full], axis=0)
        cluster_explanations = explain_clusters(
            doc_ids=[d.doc_id for d in semantic_docs],
            doc_texts=[d.raw_text for d in semantic_docs],
            doc_embeddings=doc_matrix,
            cluster_labels=cluster_result.labels,
            category_folders=[c.folder for c in categories],
            category_embeddings=cat_emb_matrix,
            member_confidences=[
                p.overall_confidence for p in semantic_packages],
        )
    unclassified = explain_unclassified(packages=packages, threshold=threshold)

    # ── 결과 출력 ───────────────────────────────────
    print("\n=== 분류 결과 (카테고리별) ===")
    grouped: dict[str, list[EvidencePackage]] = {c.folder: [] for c in categories}
    grouped["(LLM 위임)"] = []
    for pkg in packages:
        key = pkg.final_category if not pkg.needs_llm else "(LLM 위임)"
        grouped.setdefault(key, []).append(pkg)

    for folder, items in grouped.items():
        if not items:
            continue
        print(f"\n[{folder}]  ({len(items)}개)")
        for pkg in items:
            tag = "noise" if pkg.cluster.is_noise else f"cluster {pkg.cluster.label}"
            print(
                f"  - {pkg.doc_id[:60]:60s}  "
                f"conf={pkg.overall_confidence:.3f}  ({tag})"
            )

    print("\n=== 군집별 형성 이유 ===")
    for ce in cluster_explanations:
        print()
        print(ce.summary())

    if unclassified:
        print(f"\n=== 미분류 문서 사유 ({len(unclassified)}건) ===")
        for u in unclassified:
            print()
            print(u.explain())

    # ── 옵션: 문서별 근거 설명 ─────────────────────
    if explain:
        print("\n=== 문서별 분류 근거 ===")
        for pkg in packages:
            print("─" * 60)
            print(pkg.explain())

    # ── 옵션: 디스크 dump ───────────────────────────
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as f:
            for pkg in packages:
                f.write(pkg.to_json(include_vector=False) + "\n")
        print(f"\n[dump] {len(packages)}건 → {dump_path}")

    # ── 옵션: 실제 폴더 분류 ──────────────────────
    if output_root is not None:
        # packages 순서와 일치하는 docs 리스트 만들기
        doc_by_id = {d.doc_id: d for d in docs}
        ordered_docs = [doc_by_id[p.doc_id] for p in packages]
        with timer.measure("STAGE 폴더 복사"):
            layout = copy_to_classified_folders(
                output_root, categories, packages, ordered_docs)
        print(f"\n[output] {output_root} 에 카테고리 폴더 생성 및 파일 복사 완료")
        for folder, ids in layout.items():
            if ids:
                print(f"  {folder}: {len(ids)}개")

    # ── 타이밍 리포트 ───────────────────────────────
    timing_report = timer.report()
    print("\n" + timing_report)

    if timing_json_path is not None:
        timing_json_path.parent.mkdir(parents=True, exist_ok=True)
        with timing_json_path.open("w", encoding="utf-8") as f:
            json.dump(timer.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[timing] → {timing_json_path}")

    # ── 옵션: 마크다운 보고서 ─────────────────────
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown_report(
            report_path=report_path,
            categories=categories,
            packages=packages,
            grouped=grouped,
            cluster_explanations=cluster_explanations,
            unclassified=unclassified,
            timing_report=timing_report,
            output_root=output_root,
            docs=docs,
        )
        print(f"[report] → {report_path}")

    # [우선순위3] 임베딩 캐시 저장 (다음 실행 가속)
    if emb_cache is not None:
        emb_cache.save()

    return packages


def _write_markdown_report(
    *,
    report_path: Path,
    categories: list[Category],
    packages: list[EvidencePackage],
    grouped: dict[str, list[EvidencePackage]],
    cluster_explanations,
    unclassified,
    timing_report: str,
    output_root: Path | None,
    docs: list[InputDoc],
) -> None:
    lines: list[str] = []
    lines.append("# 의미기반 분류 결과 보고서")
    lines.append("")
    lines.append(f"- 입력 문서 수: **{len(packages)}건**")
    lines.append(f"- 카테고리: {len(categories)}개")
    if output_root:
        lines.append(f"- 출력 폴더: `{output_root}`")
    lines.append("")

    # 카테고리별 결과
    lines.append("## 카테고리별 분류 결과")
    for folder, items in grouped.items():
        if not items:
            continue
        lines.append(f"\n### {folder}  ({len(items)}건)")
        lines.append("")
        lines.append("| 문서 | 신뢰도 | 군집 | 1순위 후보 |")
        lines.append("| --- | --- | --- | --- |")
        for pkg in items:
            tag = "noise" if pkg.cluster.is_noise else f"c{pkg.cluster.label}"
            top1 = pkg.similarity.top_k_categories[0] if pkg.similarity.top_k_categories else ("-", 0.0)
            lines.append(
                f"| {pkg.doc_id} | {pkg.overall_confidence:.3f} | {tag} "
                f"| {top1[0]} ({top1[1]:.3f}) |"
            )

    # 군집 형성 이유
    lines.append("\n## 군집별 형성 이유")
    for ce in cluster_explanations:
        lines.append("")
        if ce.is_noise:
            lines.append(f"### 노이즈 (-1) — {ce.size}건")
            lines.append("HDBSCAN이 어떤 군집에도 묶을 만큼 밀집된 이웃을 찾지 못함.")
        else:
            lines.append(f"### 군집 {ce.cluster_label} — {ce.size}건 → **{ce.mapped_category}**")
            lines.append(f"- centroid → 카테고리 매칭 cosine: **{ce.mapped_confidence:.3f}**")
            lines.append(f"- 평균 멤버 신뢰도: {ce.avg_member_confidence:.3f}")
            kw = ", ".join(f"`{w}`" for w, _ in ce.top_keywords)
            lines.append(f"- 공통 키워드: {kw}")
            lines.append("- 대표 문서:")
            for rid in ce.representative_doc_ids:
                lines.append(f"  - {rid}")

    # 미분류 사유
    if unclassified:
        lines.append("\n## 미분류 / LLM 위임 문서 사유")
        for u in unclassified:
            lines.append("")
            lines.append(f"### {u.doc_id}")
            lines.append(f"- **사유**: {u.reason}")
            lines.append(
                f"- 신뢰도: {u.overall_confidence:.3f} (임계값 {u.threshold:.2f})"
            )
            cands = ", ".join(f"{c} ({s:.3f})" for c, s in u.top_candidates)
            lines.append(f"- top-3 후보: {cands}")

    # 단계별 시간
    lines.append("\n## 단계별 실행 시간")
    lines.append("")
    lines.append("```")
    lines.append(timing_report)
    lines.append("```")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="의미기반 분류 파이프라인")
    parser.add_argument("--input", type=Path, help="JSONL 파일 경로")
    parser.add_argument("--input-dir", type=Path, help="문서 디렉터리 경로")
    parser.add_argument("--demo", action="store_true", help="내장 데모 데이터로 실행")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="분류 결과를 복사할 출력 폴더 (카테고리별 하위 폴더 자동 생성)",
    )
    parser.add_argument(
        "--categories", type=Path, default=None,
        help="카테고리 JSON 경로 (기본: 스크립트 옆 categories.json)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="cosine similarity 확정 임계값 (기본 0.75)",
    )
    parser.add_argument("--explain", action="store_true", help="문서별 분류 근거 출력")
    parser.add_argument(
        "--dump", type=Path, default=None,
        help="EvidencePackage를 JSONL로 저장할 경로",
    )
    parser.add_argument(
        "--timing-json", type=Path, default=None,
        help="단계별 시간 측정 결과를 JSON으로 저장할 경로",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="결과 마크다운 보고서를 저장할 경로",
    )
    parser.add_argument(
        "--seed-dir", type=Path, default=None,
        help="정답지(또는 사전 분류) 디렉터리. 카테고리별 N개를 시드로 주입",
    )
    parser.add_argument(
        "--seed-per-category", type=int, default=5,
        help="카테고리당 시드 파일 수 (기본 5) [우선순위3]",
    )
    parser.add_argument(
        "--no-description-seed", action="store_true",
        help="categories.json description 기반 cold-start 시드 비활성화 "
             "(정답지 시드만 사용)",
    )
    parser.add_argument(
        "--no-rule-stage", action="store_true",
        help="STAGE A (파일명 룰) 비활성화",
    )
    parser.add_argument(
        "--conservative-rules", action="store_true",
        help="Test.zip-specific 키워드를 제외한 보수적 룰셋 사용 "
             "(overfitting 진단)",
    )
    parser.add_argument(
        "--enable-clustering", action="store_true",
        help="UMAP + HDBSCAN 단계 활성화 (기본 OFF — 시간 절약)",
    )
    parser.add_argument(
        "--cascade-low", type=float, default=None,
        help="임계값 캐스케이드 하한 (기본: threshold * 0.75)",
    )
    parser.add_argument(
        "--legacy-cosine", action="store_true",
        help="[우선순위1] 4신호 융합 대신 구 cosine-max 단일 방식 사용 (ablation용)",
    )
    parser.add_argument(
        "--semantic-threshold", type=float, default=None,
        help=f"[우선순위1] 융합 점수 확정 임계값 (기본 {SEMANTIC_DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--knn-k", type=int, default=DEFAULT_KNN_K,
        help=f"[우선순위1] k-NN 다수결 이웃 수 (기본 {DEFAULT_KNN_K})",
    )
    parser.add_argument(
        "--online-learning", action="store_true",
        help="분류 중 자동 예측 라벨을 이력에 추가 (기본 OFF — cold-start에서 "
             "오분류 전파로 정확도 하락). production 사용자확정 라벨용과는 별개",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="[우선순위3] 임베딩 디스크 캐시 비활성화",
    )
    parser.add_argument(
        "--cache-path", type=Path, default=None,
        help="[우선순위3] 임베딩 캐시 파일 경로 (기본: 출력폴더/_emb_cache.pkl)",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG 레벨 로그")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    categories = load_categories(args.categories)
    print(f"[INIT] 카테고리 {len(categories)}개 로드 — {[c.folder for c in categories]}")
    mode = "4신호 융합(BM25+centroid+kNN)" if not args.legacy_cosine else "cosine-max 단일"
    print(f"[INIT] 의미 코어: {mode} | 임베딩 {EMBED_DIM}d")

    if args.demo:
        docs = demo_documents()
    elif args.input_dir:
        print(f"[INIT] 디렉터리 로딩: {args.input_dir}")
        docs = load_directory(args.input_dir)
        print(f"[INIT] {len(docs)}개 파일 읽음")
    elif args.input:
        docs = load_jsonl(args.input)
    else:
        parser.error("--input / --input-dir / --demo 중 하나를 지정하세요.")

    run(
        docs=docs,
        categories=categories,
        threshold=args.threshold,
        explain=args.explain,
        dump_path=args.dump,
        timing_json_path=args.timing_json,
        output_root=args.output_dir,
        report_path=args.report,
        seed_dir=args.seed_dir,
        seed_per_category=args.seed_per_category,
        use_description_seed=not args.no_description_seed,
        enable_clustering=args.enable_clustering,
        cascade_low_threshold=args.cascade_low,
        use_rule_stage=not args.no_rule_stage,
        conservative_rules=args.conservative_rules,
        use_semantic_core=not args.legacy_cosine,
        semantic_threshold=args.semantic_threshold,
        knn_k=args.knn_k,
        cache_path=args.cache_path,
        use_cache=not args.no_cache,
        online_learning=args.online_learning,
    )


if __name__ == "__main__":
    main()
