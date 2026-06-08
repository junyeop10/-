# Version 2 Document Intelligence Architecture

## Direction

Version 2 moves the project from fixed-category file sorting toward a document intelligence system.

The core idea is:

- the program extracts reusable document features first
- ML, clustering, and rules make decisions from those features
- LLMs are optional assistants for ambiguous cases, naming, and explanations
- users only review uncertain documents and their corrections become training data

This is not intended to be a system where an AI model reads every whole document on every run.

## Goals

- Automatically classify document type.
- Group similar documents.
- Recommend new category or topic candidates.
- Learn gradually from user corrections.
- Strengthen classification with structural document features.
- Run realistically on CPU-first local machines.

## Pipeline

```text
File discovery
  -> text and OCR extraction
  -> feature extraction
  -> feature cache
  -> type classification
  -> tag and topic scoring
  -> similarity grouping
  -> new category candidate detection
  -> confidence and review routing
  -> user feedback
  -> retraining and rule suggestion
```

## Feature Extraction

Version 2 should treat file names, metadata, text samples, and document structure as first-class signals.

Important feature groups:

- file name, title, extension, and path hints
- page count, slide count, sheet count, file size
- abstract and references presence
- citation patterns such as `[1]`, `et al.`, `doi`, and arXiv-like identifiers
- bullet ratio and average sentence length
- table count and image count
- first-page title structure
- last-page references structure
- document-type patterns such as `제1조`, `갑`, `을`, `승인번호`, and `사업자번호`

File names should have a dedicated scoring and vectorization channel. Names such as `근로계약서_김OO.pdf`, `캡스톤_최종발표자료.pptx`, and `Transformer_MRI_Review_Paper.pdf` are strong classification evidence even when OCR output is noisy.

## Responsibility Split

### Rule-Based

Rules should detect high-precision structural and domain signals:

- contract clauses and party markers
- receipt, invoice, and business registration number patterns
- academic paper markers
- presentation markers
- strong filename hints
- exclusion and negative patterns

Rules are best for explainable, high-confidence signals.

### ML

ML should own stable document type classification:

- paper
- report
- contract
- presentation
- receipt
- miscellaneous

Recommended CPU-friendly models:

- `LinearSVC` with calibrated probabilities
- `LogisticRegression`
- `RandomForestClassifier`
- `HistGradientBoostingClassifier`

The model input should combine filename vectors, sampled text TF-IDF, structural features, metadata features, and rule-derived scores.

### Active Learning

Active learning should decide what the user needs to review:

- highest confidence is below threshold
- top-1 and top-2 score margin is too small
- rules, ML, and embedding disagree
- document belongs to a new cluster with low similarity to known categories

The first implementation can use internal uncertainty rules. A framework such as `small-text` can be added later when the training dataset is large enough.

### Clustering

Clustering should not force every document into an existing category.

Use clustering for:

- documents classified as miscellaneous
- review-required documents
- subtopics inside a known type, such as papers
- repeated new groups that may deserve a category or tag

CPU-friendly starting points:

- Generate or reuse document embeddings first.
- Run `HDBSCAN` on those embedding vectors for new-group discovery.
- Keep DBSCAN or token-bucket fallback only for environments where optional clustering dependencies are unavailable.

### LLM Assistance

LLMs should be optional and limited:

- suggest names for new clusters
- summarize why a category candidate exists
- propose tags for a small representative sample
- help with ambiguous documents only after cheaper signals disagree

The default system should work without sending every document to an LLM.

## Data Model Additions

Recommended storage additions:

```text
document_features
  file_id
  feature_version
  filename_features_json
  metadata_features_json
  structural_features_json
  text_stats_json
  created_at

document_vectors
  file_id
  vector_type
  vector_key
  model_version
  created_at

model_runs
  model_name
  model_version
  trained_at
  training_count
  metrics_json

category_candidates
  candidate_id
  source
  suggested_name
  representative_file_ids_json
  evidence_json
  status

document_tags
  file_id
  tag
  confidence
  source
  created_at
```

These tables keep type, tag, cluster, and candidate-category behavior separate.

## Classification Output

Version 2 should prefer this shape:

```text
type: 논문
type_confidence: 0.91
tags:
  - AI: 0.88
  - 의료: 0.74
  - 컴퓨터비전: 0.69
review_required: false
explanation:
  - filename matched review paper pattern
  - abstract and references detected
  - citation density is high
  - nearest confirmed examples are paper documents
```

Single forced category assignment should not be the only output.

## CPU-First Strategy

- Cache OCR output by file hash and extractor version.
- Cache feature extraction by file hash and feature version.
- Cache embeddings by text signature and model version.
- Use sampled text instead of whole-document embeddings.
- Batch embedding generation when possible.
- Prefer sparse TF-IDF models for the main classifier.
- Use HDBSCAN clustering on embedded subsets rather than the full corpus every time.

## Main Risks

- Score blending can become hard to tune as more signals are added.
- OCR and embedding inference can dominate runtime without strong caching.
- New category detection can create noisy categories if every outlier is promoted.
- Feedback that only creates token boosts will not be enough for long-term learning.
- Type and tag concepts can become mixed unless stored separately.
- Clustering quality depends heavily on feature quality and representative sampling.

## Recommended Implementation Order

1. Add a document feature extractor and cache.
2. Store filename, metadata, structural, and text-stat features.
3. Add a CPU-friendly type classifier trained from user-confirmed examples.
4. Upgrade review routing with confidence, margin, and source-conflict checks.
5. Add tag storage separate from type/category.
6. Generate embeddings for miscellaneous and review-required documents, then run HDBSCAN clustering on those vectors.
7. Add category candidate review and accept/reject flow.
8. Feed accepted corrections into retraining and rule suggestion.

## Relationship to Existing System

The current project already has useful foundations:

- hybrid rule, embedding, metadata, feedback, and duplicate scoring
- OCR cache and embedding cache
- feedback logs and confirmed examples
- hierarchical taxonomy fields
- explainable result payloads
- preview-first file movement and recovery safety

Version 2 should extend those foundations instead of replacing them.
