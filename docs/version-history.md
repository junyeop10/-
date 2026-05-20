# Version History

This document is the presentation-friendly version summary for the project.

## v0.8.0 - Interactive GUI operations and performance observability

- Added persistent embedding cache observability with cache-hit metadata.
- Added GUI category-tree browsing with folder-like grouped results.
- Added in-GUI drag-and-drop recategorization between category folders.
- Added GUI panels for move preview/history, feedback log management, embedding cache management, and performance analysis.
- Added startup timing and per-file latency analysis with visible bottleneck explanations.
- Preserved expanded category state during GUI recategorization refreshes.
- Fixed feedback-log deletion so linked confirmed-example rows are removed safely.

## v0.7.0 - Enterprise MVP safety and operations foundation

- Added hierarchical category support with versioned taxonomy/config loading.
- Added richer classification persistence with explanation payloads and source score breakdowns.
- Added preview-first move operations, restore commands, and recovery snapshots.
- Added feedback log management and rebuildable adaptive learning.
- Added operator/developer documentation for architecture, safety, and recovery.

## v0.6.0 - Input format expansion and multilingual rule coverage

- Added `docx`, `xlsx`, and `pptx` readers.
- Extended the pipeline beyond `txt` and `pdf`.
- Added English keyword coverage on top of Korean rules.

## v0.5.0 - GUI usability and background preparation

- Improved GUI startup behavior.
- Added OCR-used visibility in the result list.
- Kept the main screen responsive while background preparation handles heavier stages.

## v0.4.1 - OCR optimization and filename-based evidence

- Reduced OCR usage by skipping files that can already be classified from filename hints.
- Added configurable OCR worker count and minimum-text thresholds.
- Reused OCR engine instances per process.

## v0.4.0 - OCR fallback for scanned PDFs

- Added OCR fallback for text-poor PDFs.
- Limited OCR to targeted cases rather than all PDFs.
- Recorded OCR usage in explanations and evidence.

## v0.3.0 - Local LLM support for ambiguous documents

- Added Ollama-backed `qwen2.5:3b` as an optional ambiguity resolver.
- Only low-confidence documents route to LLM.
- Preserved rule/embedding output when LLM is unavailable.

## v0.2.0 - Rule restoration and accuracy recovery

- Restored Korean rule/category coverage.
- Improved context rules and category keywords.
- Strengthened document-type handling for certificates and corporate records.

## v0.1.0 - GitHub-ready baseline

- Cleaned the repository for versioned development.
- Added documentation and ignore rules.
- Established the first reproducible project baseline.
