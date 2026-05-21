"""Category profile seeding and synthetic training row generation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.text_cleaner import tokenize_text


DEFAULT_CATEGORY_PROFILES: list[dict[str, Any]] = [
    {
        "type": "계약서",
        "profile_text": (
            "계약서는 당사자 간 권리와 의무를 정의하는 문서다. "
            "주로 제1조, 제2조 같은 조항 구조를 가지며, 갑/을 표현, 계약기간, 손해배상, 비밀유지, 계약 해지 등의 표현이 자주 등장한다. "
            "문단 중심 구조이며 긴 문장과 법률 표현이 많다. "
            "표보다는 텍스트 비중이 높고 조항 번호 구조가 반복된다."
        ),
        "tags": ["법률", "계약", "조항"],
    },
    {
        "type": "영수증",
        "profile_text": (
            "영수증 또는 매출전표는 결제 사실과 금액 정보를 기록하는 문서다. "
            "승인번호, 카드번호, 결제일시, 합계, 공급가액, 부가세 같은 표현이 자주 등장한다. "
            "숫자와 금액 비율이 높고 짧은 줄이 반복된다. "
            "세로로 긴 레이아웃이 많으며 가격 정보가 반복된다."
        ),
        "tags": ["결제", "금액", "영수증"],
    },
    {
        "type": "발표자료",
        "profile_text": (
            "발표자료는 핵심 내용을 요약하여 전달하는 문서다. "
            "큰 제목과 짧은 bullet 문장이 많고 이미지와 도표 비율이 높다. "
            "문장 길이가 짧고 페이지당 텍스트 양이 적은 경우가 많다. "
            "슬라이드 구조로 구성되며 핵심 키워드 위주 표현이 반복된다."
        ),
        "tags": ["슬라이드", "발표", "요약"],
    },
    {
        "type": "세금계산서",
        "profile_text": (
            "세금계산서 또는 인보이스는 거래 품목과 금액 정보를 기록하는 문서다. "
            "공급자, 공급받는자, 품목, 단가, 수량, 합계금액, 사업자등록번호 등의 표현이 자주 등장한다. "
            "표 구조와 금액 열이 반복되며 숫자와 표 비율이 높다."
        ),
        "tags": ["세금", "계산서", "거래"],
    },
    {
        "type": "논문",
        "profile_text": (
            "논문 또는 기술 연구 문서는 연구 목적과 결과를 설명하는 문서다. "
            "abstract, references, DOI, citation 같은 표현이 자주 등장한다. "
            "텍스트 밀도가 높고 표와 그래프가 포함되는 경우가 많다. "
            "2단 구조나 긴 문단 구조를 가지며 참고문헌 영역이 마지막에 위치하는 경우가 많다."
        ),
        "tags": ["연구", "논문", "기술"],
    },
]


def build_synthetic_training_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one active category profile into weighted synthetic training rows."""
    category_type = str(profile.get("type", "")).strip()
    profile_text = str(profile.get("profile_text", "")).strip()
    if not category_type or not profile_text:
        return []
    synthetic_count = max(1, int(profile.get("synthetic_count", 5) or 5))
    weight = float(profile.get("weight", 0.5) or 0.5)
    tags = profile.get("tags", [])
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except json.JSONDecodeError:
            tags = []
    terms = _synthetic_terms(profile_text)
    rows: list[dict[str, Any]] = []
    for index in range(synthetic_count):
        sampled_terms = terms[index::synthetic_count] or terms[:8]
        synthetic_text = "\n".join(
            [
                "# TYPE",
                category_type,
                "# PROFILE",
                profile_text,
                "# SYNTHETIC_TERMS",
                "\n".join(sampled_terms[:12]),
                "# TAGS",
                ", ".join(str(tag) for tag in tags),
            ]
        )
        rows.append(
            {
                "label": category_type,
                "file_name": f"synthetic_{category_type}_{index + 1}.txt",
                "body_text": synthetic_text,
                "structural_features": {},
                "layout_features": {},
                "source": "category_profile",
                "source_id": profile.get("id"),
                "sample_weight": weight,
            }
        )
    return rows


def build_category_profile_signature(profiles: list[dict[str, Any]]) -> str:
    active_profiles = [profile for profile in profiles if str(profile.get("status", "active")) == "active"]
    payload = {
        "active_profile_count": len(active_profiles),
        "profiles": [
            {
                "id": profile.get("id"),
                "type": profile.get("type"),
                "updated_at": profile.get("updated_at"),
                "profile_text_hash": hashlib.sha256(str(profile.get("profile_text", "")).encode("utf-8")).hexdigest(),
                "synthetic_count": int(profile.get("synthetic_count", 0) or 0),
                "weight": float(profile.get("weight", 0.0) or 0.0),
            }
            for profile in active_profiles
        ],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _synthetic_terms(profile_text: str) -> list[str]:
    tokens = tokenize_text(profile_text)
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms[:80]
