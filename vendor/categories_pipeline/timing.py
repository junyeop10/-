"""
timing.py
---------
파이프라인 단계별 실행 시간 측정 모듈.

목적
----
4단계 파이프라인(임베딩 → 차원 축소 → HDBSCAN → cosine similarity)이
각각 얼마나 걸리는지, 그리고 같은 단계 안에서 문서별 처리에 평균 얼마나
드는지를 한눈에 볼 수 있게 한다.

핵심 개념
---------
- ``measure(name)`` : top-level 단계로 기록 (TOTAL 비중 계산에 포함)
- ``measure(name, sub=True)`` : 하위 작업으로 기록 (TOTAL 비중에서는 제외)
  ※ STEP 1 안에서 문서별 임베딩을 따로 잴 때 중복 합산 방지용

사용 예시
---------
    timer = StepTimer()

    with timer.measure("STEP 1: 임베딩"):
        for d in docs:
            with timer.measure("  embed per doc", sub=True):
                embed(d)

    with timer.measure("STEP 2: 차원 축소"):
        reduce(vectors)

    print(timer.report())
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class StepStats:
    name: str
    count: int
    total: float    # 초
    mean: float
    min: float
    max: float
    is_sub: bool    # True면 TOTAL 비중 계산에서 제외


# ──────────────────────────────────────────────
# StepTimer
# ──────────────────────────────────────────────

class StepTimer:
    """이름이 붙은 구간의 실행 시간을 누적 기록한다."""

    def __init__(self) -> None:
        # name → list[float] (각 호출의 경과 시간)
        self._records: dict[str, list[float]] = {}
        # 처음 등록된 순서대로 리포트 출력
        self._order: list[str] = []
        # name → is_sub (TOTAL 비중에서 제외할지 여부)
        self._sub_flags: dict[str, bool] = {}

    # ── 측정 ──────────────────────────────────

    @contextmanager
    def measure(self, name: str, sub: bool = False) -> Iterator[None]:
        """구간 실행 시간을 기록한다.

        Parameters
        ----------
        name : 구간 이름. 같은 이름을 여러 번 호출하면 누적 평균/합계가 잡힌다.
        sub  : True면 하위 작업으로 표시 (TOTAL 합산에서 제외).
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._record(name, elapsed, sub=sub)

    def _record(self, name: str, elapsed: float, sub: bool) -> None:
        if name not in self._records:
            self._records[name] = []
            self._order.append(name)
            self._sub_flags[name] = sub
        self._records[name].append(elapsed)

    # ── 집계 ──────────────────────────────────

    def stats(self) -> list[StepStats]:
        out: list[StepStats] = []
        for name in self._order:
            times = self._records[name]
            if not times:
                continue
            out.append(StepStats(
                name=name,
                count=len(times),
                total=sum(times),
                mean=sum(times) / len(times),
                min=min(times),
                max=max(times),
                is_sub=self._sub_flags[name],
            ))
        return out

    @property
    def top_level_total(self) -> float:
        """sub=True 구간을 제외한 top-level 시간 합계."""
        return sum(
            sum(self._records[name])
            for name in self._order
            if not self._sub_flags[name]
        )

    # ── 리포트 ────────────────────────────────

    def report(self) -> str:
        rows = self.stats()
        total = self.top_level_total or 1e-12   # 0 나눗셈 방지

        lines: list[str] = []
        sep_thick = "=" * 86
        sep_thin = "-" * 86
        lines.append(sep_thick)
        lines.append("Timing Report")
        lines.append(sep_thick)
        lines.append(
            f"{'Step':40s}  {'N':>4s}  {'Total(s)':>9s}  "
            f"{'Mean(s)':>9s}  {'Min(s)':>8s}  {'Max(s)':>8s}  {'Share':>6s}"
        )
        lines.append(sep_thin)

        for s in rows:
            share = "  -   " if s.is_sub else f"{s.total / total * 100:5.1f}%"
            min_s = f"{s.min:.4f}" if s.count > 1 else "  -   "
            max_s = f"{s.max:.4f}" if s.count > 1 else "  -   "
            lines.append(
                f"{s.name[:40]:40s}  {s.count:>4d}  {s.total:>9.4f}  "
                f"{s.mean:>9.4f}  {min_s:>8s}  {max_s:>8s}  {share:>6s}"
            )

        lines.append(sep_thin)
        lines.append(
            f"{'TOP-LEVEL TOTAL':40s}        {total:>9.4f}"
            + " " * 30
            + "100.0%"
        )
        lines.append(sep_thick)
        return "\n".join(lines)

    # ── 직렬화 ────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_level_total": self.top_level_total,
            "stats": [
                {
                    "name": s.name,
                    "count": s.count,
                    "total": s.total,
                    "mean": s.mean,
                    "min": s.min,
                    "max": s.max,
                    "is_sub": s.is_sub,
                }
                for s in self.stats()
            ],
            "raw_records": {
                name: list(times) for name, times in self._records.items()
            },
        }

    def reset(self) -> None:
        self._records.clear()
        self._order.clear()
        self._sub_flags.clear()


# ──────────────────────────────────────────────
# 데코레이터
# ──────────────────────────────────────────────

def timed(timer: StepTimer, name: str | None = None, sub: bool = False) -> Callable:
    """함수 호출마다 자동으로 timer에 시간을 기록하는 데코레이터."""
    def deco(fn: Callable) -> Callable:
        label = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timer.measure(label, sub=sub):
                return fn(*args, **kwargs)
        return wrapper
    return deco
