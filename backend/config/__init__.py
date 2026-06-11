"""
config 패키지 — 팀이 수정하는 JSON 설정

  keywords.json   본문·파일명 키워드 (Stage1, Stage3)
  categories.json RAG·LLM 카테고리 설명·예시 (Stage5)
  loader.py       위 JSON 을 읽는 로더
"""

from config.loader import BASE_KEYWORDS, reload_keywords

__all__ = ["BASE_KEYWORDS", "reload_keywords"]
