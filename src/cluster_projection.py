"""2D projection exports for inspecting document clusters."""

from __future__ import annotations

import html
import json
from collections import Counter
from typing import Any

import numpy as np

from src.vectorizer import cosine_similarity


def build_cluster_projection(
    documents: list[dict[str, Any]],
    vectors: list[list[float]],
    cluster_ids: list[int],
    *,
    probabilities: list[float] | None = None,
    neighbor_count: int = 3,
) -> dict[str, Any]:
    """Project clustering vectors to 2D and attach nearest-neighbor explanations."""
    if not documents or not vectors:
        return {"points": [], "method": "empty", "cluster_counts": {}}

    matrix = np.asarray(vectors, dtype=np.float32)
    coords, method = _project_to_2d(matrix)
    normalized_coords = _normalize_coords(coords)
    probabilities = probabilities or [0.0 for _ in documents]

    points = []
    for index, (document, vector, cluster_id) in enumerate(zip(documents, vectors, cluster_ids)):
        evidence = document.get("evidence", document)
        points.append(
            {
                "index": index,
                "filename": evidence.get("filename", document.get("filename", "")),
                "file_path": evidence.get("file_path", document.get("file_path", "")),
                "cluster_id": int(cluster_id),
                "parent_cluster_id": int(document.get("parent_cluster_id", -1)),
                "probability": round(float(probabilities[index]) if index < len(probabilities) else 0.0, 4),
                "x": round(float(normalized_coords[index][0]), 6),
                "y": round(float(normalized_coords[index][1]), 6),
                "nearest_neighbors": _nearest_neighbors(index, documents, vectors, neighbor_count=neighbor_count),
                "top_tokens": evidence.get("top_tokens", [])[:8],
                "filename_tokens": evidence.get("filename_tokens", [])[:8],
                "extraction_status": evidence.get("extraction_status", ""),
            }
        )

    cluster_counts = Counter(str(cluster_id) for cluster_id in cluster_ids)
    return {
        "method": method,
        "note": "2D projection is for visual inspection only and does not affect category classification.",
        "cluster_counts": dict(sorted(cluster_counts.items())),
        "points": points,
    }


def render_cluster_projection_html(projection: dict[str, Any]) -> str:
    """Render a small standalone HTML scatter plot."""
    points = projection.get("points", [])
    points_json = json.dumps(points, ensure_ascii=False)
    method = html.escape(str(projection.get("method", "")))
    note = html.escape(str(projection.get("note", "")))
    counts = html.escape(json.dumps(projection.get("cluster_counts", {}), ensure_ascii=False))
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Cluster Projection</title>
  <style>
    body {{ font-family: Segoe UI, Malgun Gothic, sans-serif; margin: 18px; color: #1f2933; }}
    .wrap {{ display: grid; grid-template-columns: minmax(520px, 1fr) 420px; gap: 18px; }}
    svg {{ width: 100%; height: 680px; border: 1px solid #c8ced8; background: #fbfcfe; }}
    .point {{ cursor: pointer; stroke: #222; stroke-width: 0.7; opacity: 0.86; }}
    .point.noise {{ fill: #8a8f98; }}
    .panel {{ border: 1px solid #c8ced8; padding: 12px; min-height: 220px; white-space: pre-wrap; }}
    .list {{ margin-top: 14px; max-height: 390px; overflow: auto; border-top: 1px solid #dde2ea; padding-top: 8px; }}
    button {{ margin: 3px; }}
  </style>
</head>
<body>
  <h2>Cluster Projection</h2>
  <p>method={method} | cluster_counts={counts}</p>
  <p>{note}</p>
  <div class="wrap">
    <svg id="plot" viewBox="0 0 1000 680" role="img"></svg>
    <div>
      <div id="detail" class="panel">점을 클릭하면 파일명, cluster_id, 가까운 이웃이 표시됩니다.</div>
      <div id="list" class="list"></div>
    </div>
  </div>
  <script>
    const points = {points_json};
    const colors = ["#2f80ed","#eb5757","#27ae60","#f2994a","#9b51e0","#00a6a6","#d61f69","#7a7f00","#33658a","#f26419"];
    const plot = document.getElementById("plot");
    const detail = document.getElementById("detail");
    const list = document.getElementById("list");
    function colorFor(clusterId) {{
      if (clusterId === -1) return "#8a8f98";
      return colors[Math.abs(clusterId) % colors.length];
    }}
    function showPoint(point) {{
      const neighbors = (point.nearest_neighbors || []).map(n => `- ${{n.filename}} | cosine=${{n.cosine_similarity}} | cluster=${{n.cluster_id}}`).join("\\n");
      const tokens = (point.top_tokens || []).map(t => t.token || "").filter(Boolean).join(", ");
      detail.textContent = `파일: ${{point.filename}}\\nparent_cluster_id: ${{point.parent_cluster_id}}\\nfine_cluster_id: ${{point.cluster_id}}\\nprobability: ${{point.probability}}\\n좌표: (${{point.x}}, ${{point.y}})\\n상태: ${{point.extraction_status}}\\ntop_tokens: ${{tokens}}\\n\\n가까운 이웃:\\n${{neighbors || "none"}}`;
    }}
    for (const point of points) {{
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", 40 + point.x * 920);
      circle.setAttribute("cy", 640 - point.y * 600);
      circle.setAttribute("r", point.cluster_id === -1 ? 5 : 7);
      circle.setAttribute("fill", colorFor(point.cluster_id));
      circle.setAttribute("class", point.cluster_id === -1 ? "point noise" : "point");
      circle.addEventListener("click", () => showPoint(point));
      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      title.textContent = `${{point.filename}} | cluster ${{point.cluster_id}}`;
      circle.appendChild(title);
      plot.appendChild(circle);
    }}
    list.innerHTML = points.map(p => `<button style="border-left:12px solid ${{colorFor(p.cluster_id)}}">${{p.cluster_id}} | ${{p.filename}}</button>`).join("");
    [...list.querySelectorAll("button")].forEach((button, index) => button.addEventListener("click", () => showPoint(points[index])));
  </script>
</body>
</html>
"""


def _project_to_2d(matrix: np.ndarray) -> tuple[np.ndarray, str]:
    if matrix.shape[0] == 1:
        return np.asarray([[0.5, 0.5]], dtype=np.float32), "single-point"
    if matrix.shape[1] <= 2:
        if matrix.shape[1] == 1:
            return np.column_stack([matrix[:, 0], np.zeros(matrix.shape[0])]), "raw-vector"
        return matrix[:, :2], "raw-vector"
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(matrix), "pca-2d"
    except Exception:
        return matrix[:, :2], "first-two-dimensions"


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    result = coords.astype(np.float32, copy=True)
    for axis in (0, 1):
        values = result[:, axis]
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if max_value - min_value <= 1e-12:
            result[:, axis] = 0.5
        else:
            result[:, axis] = (values - min_value) / (max_value - min_value)
    return result


def _nearest_neighbors(
    index: int,
    documents: list[dict[str, Any]],
    vectors: list[list[float]],
    *,
    neighbor_count: int,
) -> list[dict[str, Any]]:
    scored = []
    source = vectors[index]
    for other_index, vector in enumerate(vectors):
        if other_index == index:
            continue
        score = cosine_similarity(source, vector)
        other_doc = documents[other_index]
        evidence = other_doc.get("evidence", other_doc)
        scored.append(
            {
                "filename": evidence.get("filename", other_doc.get("filename", "")),
                "cluster_id": int(other_doc.get("cluster_id", -1)),
                "cosine_similarity": round(float(score), 4),
            }
        )
    return sorted(scored, key=lambda item: item["cosine_similarity"], reverse=True)[:neighbor_count]
