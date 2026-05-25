from models.schemas import ClassifyResult


def run(results: list[ClassifyResult]) -> list[ClassifyResult]:
    """Stage 2 (HDBSCAN) is not implemented; pass through to stage 3."""
    return results
