"""
lexical.py
----------
BM25 어휘(lexical) 매칭 — 의미기반 코어의 *보조 신호*.

왜 필요한가
-----------
임베딩(cosine)은 "의미"는 잘 잡지만, 한국어 비즈니스 문서에서 결정적인
*정확한 단어*("세금계산서", "법인등기부등본", "제안요청서")의 영향력을
희석시킨다. 임베딩 공간에서 "견적서"와 "계약서"는 매우 가깝지만,
어휘 매칭은 둘을 또렷이 구분한다.

BM25는 그 정확한 단어 매칭을 점수화한다. 단독으로 쓰지 않고 cosine 신호와
함께 융합해(semantic_core.py) "의미 + 어휘"를 같이 본다.

이것은 룰이 아니다
------------------
룰은 사람이 키워드를 손으로 적어 카테고리에 못박는다. BM25는 *시드/피드백
문서 본문에서 단어 분포를 데이터로 학습*해 유사도를 계산한다. 새 카테고리·
새 문서가 들어와도 코퍼스만 갱신하면 자동 반영 — 즉 의미기반 코어의 일부
(데이터 주도)이지 하드코딩 룰이 아니다.

표준 Okapi BM25
---------------
    idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score(Q,D) = Σ_{t∈Q} idf(t) · f(t,D)·(k1+1)
                              ───────────────────────────────────
                              f(t,D) + k1·(1 - b + b·|D|/avgdl)
k1=1.5, b=0.75 (관용적 기본값).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


# 한국어 2자+ 음절 / 영어 3자+ 단어 / 숫자 4자+ 토큰.
_TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{3,}|[0-9]{4,}")

# 의미가 거의 없는 흔한 토큰만 최소로 제거 (과하게 자르면 BM25 신호가 죽음).
_STOPWORDS = {
    "그리고", "있는", "있다", "하는", "위해", "통해", "관련", "대한",
    "위한", "이는", "그것", "이를", "이상", "이하", "기준", "경우",
    "또는", "the", "and", "for", "with", "this", "that", "from",
    "are", "was", "you", "have", "has", "pdf", "hwp", "hwpx",
    "docx", "xlsx", "ver", "vol",
}


def tokenize(text: str) -> list[str]:
    """한국어/영어/숫자 토큰화 + stopword 제거. 전부 소문자 정규화."""
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


@dataclass
class BM25Index:
    """시드/피드백 문서 코퍼스에 대한 BM25 인덱스.

    SemanticStore와 행(row)이 1:1로 대응한다 — i번째 doc은 i번째 저장 벡터와
    같은 문서. classify 시 쿼리 텍스트를 넣어 행별 BM25 점수를 얻는다.
    """

    k1: float = 1.5
    b: float = 0.75

    _docs_tokens: list[list[str]] = field(default_factory=list)
    _doc_freqs: list[Counter] = field(default_factory=list)
    _doc_len: list[int] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _avgdl: float = 0.0

    def add(self, text: str) -> None:
        toks = tokenize(text)
        tf = Counter(toks)
        self._docs_tokens.append(toks)
        self._doc_freqs.append(tf)
        self._doc_len.append(len(toks))
        for term in tf.keys():
            self._df[term] += 1
        n = len(self._doc_len)
        self._avgdl = sum(self._doc_len) / n if n else 0.0

    def add_batch(self, texts: list[str]) -> None:
        for t in texts:
            self.add(t)

    @property
    def n_docs(self) -> int:
        return len(self._docs_tokens)

    def _idf(self, term: str) -> float:
        n = self.n_docs
        df = self._df.get(term, 0)
        return max(0.0, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    def scores(self, query_text: str) -> list[float]:
        """쿼리에 대한 각 코퍼스 문서의 BM25 점수 (행별, 길이 n_docs)."""
        n = self.n_docs
        if n == 0:
            return []
        q_terms = set(tokenize(query_text))
        if not q_terms:
            return [0.0] * n
        idf_cache = {t: self._idf(t) for t in q_terms}
        out: list[float] = []
        for i in range(n):
            tf = self._doc_freqs[i]
            dl = self._doc_len[i]
            denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl if self._avgdl else 1.0))
            s = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if f:
                    s += idf_cache[t] * (f * (self.k1 + 1.0)) / (f + denom_norm)
            out.append(s)
        return out


def normalize_scores(scores: list[float]) -> list[float]:
    """BM25 raw 점수를 0~1로 정규화 (max로 나눔, 모두 0이면 0)."""
    if not scores:
        return []
    hi = max(scores)
    if hi <= 1e-9:
        return [0.0] * len(scores)
    return [s / hi for s in scores]
