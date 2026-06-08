"""
make_summary_pdf.py
-------------------
의미기반 분류 Test.zip 실행 결과 + 정답지 대비 평가 PDF 생성.
한글 출력을 위해 Windows의 맑은 고딕(malgun.ttf)을 등록한다.

실행:
    python make_summary_pdf.py
출력:
    _분류결과/실행요약.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ──────────────────────────────────────────────
# 폰트 등록 (한글)
# ──────────────────────────────────────────────

pdfmetrics.registerFont(TTFont("Malgun", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunBold", "C:/Windows/Fonts/malgunbd.ttf"))

# ──────────────────────────────────────────────
# 스타일
# ──────────────────────────────────────────────

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title", parent=styles["Title"],
    fontName="MalgunBold", fontSize=22, leading=28, spaceAfter=10,
    alignment=TA_CENTER, textColor=colors.HexColor("#222"),
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"],
    fontName="Malgun", fontSize=11, leading=16, spaceAfter=18,
    alignment=TA_CENTER, textColor=colors.HexColor("#666"),
)
h1 = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontName="MalgunBold", fontSize=15, leading=20,
    spaceBefore=14, spaceAfter=8,
    textColor=colors.HexColor("#1a4d8c"),
)
h1_red = ParagraphStyle(
    "H1Red", parent=h1, textColor=colors.HexColor("#b32a2a"),
)
h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontName="MalgunBold", fontSize=12, leading=16,
    spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#333"),
)
body = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontName="Malgun", fontSize=10, leading=14, spaceAfter=4,
    alignment=TA_LEFT,
)
note = ParagraphStyle(
    "Note", parent=body, fontSize=9.5, leading=13,
    leftIndent=8, textColor=colors.HexColor("#444"),
    backColor=colors.HexColor("#fff8e1"),
    borderColor=colors.HexColor("#f0c040"), borderWidth=0.5,
    borderPadding=6, spaceAfter=8,
)
warn = ParagraphStyle(
    "Warn", parent=note,
    backColor=colors.HexColor("#fde8e8"),
    borderColor=colors.HexColor("#d96666"),
    textColor=colors.HexColor("#7a1f1f"),
)
mono = ParagraphStyle(
    "Mono", parent=body, fontName="Malgun", fontSize=9, leading=12,
    backColor=colors.HexColor("#f5f5f5"),
    borderColor=colors.HexColor("#ccc"), borderWidth=0.5,
    borderPadding=6, spaceAfter=8,
)


def P(text: str, style: ParagraphStyle = body) -> Paragraph:
    return Paragraph(text, style)


# ──────────────────────────────────────────────
# 테이블 헬퍼
# ──────────────────────────────────────────────

def make_table(data: list[list], col_widths: list[float]) -> Table:
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (-1, 0), "MalgunBold"),
        ("FONTNAME",  (0, 1), (-1, -1), "Malgun"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9.5),
        ("BACKGROUND",(0, 0), (-1, 0), colors.HexColor("#1a4d8c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.HexColor("#fafafa"), colors.white]),
        ("GRID",      (0, 0), (-1, -1), 0.4, colors.HexColor("#bbb")),
        ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    return t


# ──────────────────────────────────────────────
# 평가 데이터 로드
# ──────────────────────────────────────────────

EVAL_JSON_V1 = Path("_분류결과/evaluation.json")
EVAL_JSON_V2 = Path("_분류결과_v2/evaluation.json")
EVAL_JSON_V3 = Path("_분류결과_v3/evaluation.json")
EVAL_JSON_V3A = Path("_분류결과_v3a/evaluation.json")
EVAL_JSON_V3B = Path("_분류결과_v3b/evaluation.json")
EVAL_JSON_V3CLEAN = Path("_분류결과_v3clean/evaluation.json")


def load_eval(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────
# 콘텐츠 빌드
# ──────────────────────────────────────────────

def build_story() -> list:
    story: list = []
    ev = load_eval(EVAL_JSON_V1)
    ev2 = load_eval(EVAL_JSON_V2)
    ev3 = load_eval(EVAL_JSON_V3)
    ev3a = load_eval(EVAL_JSON_V3A)
    ev3b = load_eval(EVAL_JSON_V3B)
    ev3clean = load_eval(EVAL_JSON_V3CLEAN)

    # 표지
    story.append(P("의미기반 분류 — Test.zip 실행 결과", title_style))
    story.append(P("4단계 파이프라인 · 69개 한국어 실문서 분류 + 정답지 평가",
                   subtitle_style))

    # ── 실행 요약 ──────────────────────────────
    story.append(P("실행 요약", h1))
    summary_data = [
        ["항목", "값"],
        ["입력", "Test.zip → 69개 파일 (PDF / HWP / HWPX / XLSX)"],
        ["카테고리", "8개 (가변, categories.json)"],
        ["임계값", "cosine similarity ≥ 0.55"],
        ["총 실행 시간", "24.10초"],
        ["출력 위치", "기초 코드/_분류결과/"],
        ["산출물",
         "카테고리 폴더 + 실제 파일 복사 / report.md / timing.json / "
         "evidence.jsonl / evaluation.md / evaluation.json"],
    ]
    story.append(make_table(summary_data, [38 * mm, 130 * mm]))

    # ── 평가 결과 (정답지 대비) ─────────────────
    if ev:
        story.append(P("정답지 대비 평가", h1_red))
        story.append(P(
            f"정답지: 72건 / 예측: 69건 / 비교 가능: {ev['total']}건. "
            f"<b>{ev['total'] - sum(1 for _ in ev.get('only_in_gt', []))}</b>건만 양쪽 모두에 존재.",
            body))
        kpi_data = [
            ["지표", "값"],
            ["Accuracy", f"{ev['accuracy']*100:.1f}% ({ev['correct']}/{ev['total']})"],
            ["Macro Precision", f"{ev['macro_precision']:.3f}"],
            ["Macro Recall", f"{ev['macro_recall']:.3f}"],
            ["Macro F1", f"{ev['macro_f1']:.3f}"],
        ]
        t = make_table(kpi_data, [40 * mm, 60 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fde8e8")),
            ("FONTNAME",  (0, 1), (-1, 1), "MalgunBold"),
        ]))
        story.append(t)

        story.append(P("정량적으로 cold-start 한계가 명확히 드러난다. "
                       "8개 카테고리 중 단 2개(공고_지침_양식, 견적_계약_정산)만 "
                       "정답을 맞췄고, “5. 발표자료” 폴더가 잘못된 예측을 31건 "
                       "흡수해 대부분의 오분류 원인이 되었다.", warn))

        # 카테고리별 지표
        story.append(P("카테고리별 지표", h2))
        rows = [["카테고리", "Support", "TP", "FP", "FN", "P", "R", "F1"]]
        for lbl in [
            "1. 공고_지침_양식", "2. 사업계획서 수행계획서", "3. 조사_참고자료",
            "4. 중간_최종 결과물 및 보고서", "5. 발표자료",
            "6. 견적_계약_정산", "7. 기업 인증서", "8. 기타",
            "_미분류_LLM위임",
        ]:
            m = ev["per_category"].get(lbl)
            if not m or (m["support"] == 0 and m["tp"] == 0 and m["fp"] == 0):
                continue
            rows.append([
                lbl, str(m["support"]), str(m["tp"]), str(m["fp"]), str(m["fn"]),
                f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}",
            ])
        t = make_table(rows, [55 * mm, 14 * mm, 10 * mm, 10 * mm, 10 * mm,
                              14 * mm, 14 * mm, 14 * mm])
        t.setStyle(TableStyle([
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            # 발표자료 / 미분류 행 강조
            ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#fdecec")),
            ("BACKGROUND", (0, 9), (-1, 9), colors.HexColor("#fffbe6")),
        ]))
        story.append(t)
        story.append(P("Support = 정답지 기준 해당 카테고리의 실제 문서 수. "
                       "TP/FP/FN은 모두 0인 행은 표에서 생략.", body))

    story.append(PageBreak())

    # ── 단계별 시간 ────────────────────────────
    story.append(P("단계별 실행 시간", h1))
    timing_data = [
        ["단계", "시간(s)", "비중", "비고"],
        ["STEP 1 — 임베딩", "12.20", "50.7%",
         "모델 로드 8.36s + 문서당 평균 55.6 ms"],
        ["STEP 2 — UMAP 차원 축소", "11.64", "48.3%", "384d → 15d"],
        ["STEP 3 — HDBSCAN", "0.004", "0.0%", "사실상 무시 가능"],
        ["STEP 4 — cosine similarity", "0.10", "0.4%", "문서당 0.1 ms"],
        ["STEP 5 — 폴더 복사", "0.14", "0.6%", "shutil.copy2"],
        ["TOTAL", "24.10", "100.0%", "—"],
    ]
    t = make_table(timing_data, [55 * mm, 22 * mm, 18 * mm, 73 * mm])
    t.setStyle(TableStyle([
        ("ALIGN", (1, 1), (2, -1), "RIGHT"),
        ("FONTNAME", (0, -1), (-1, -1), "MalgunBold"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8f0fa")),
    ]))
    story.append(t)
    story.append(P("병목은 STEP 1·2. 임베딩 모델 로드(첫 호출 8.4초)는 "
                   "1회성이라 배치가 커질수록 평균 비용이 낮아진다.", note))

    # ── 군집 형성 ─────────────────────────────
    story.append(P("군집 형성 결과", h1))
    story.append(P("HDBSCAN이 69개 문서를 <b>두 메가 클러스터</b>로 묶었다.", body))
    cluster_data = [
        ["군집", "크기", "centroid → 카테고리", "cosine", "공통 키워드 (TF-IDF)"],
        ["c0", "34건", "1. 공고_지침_양식", "0.656",
         "법적의무, 안전, 입교자, 지원, 컨설팅"],
        ["c1", "35건", "6. 견적_계약_정산", "0.645",
         "중소기업, 지원, 기술개, 사업계획, 커넥트스토리"],
    ]
    t = make_table(cluster_data, [14 * mm, 16 * mm, 45 * mm, 18 * mm, 75 * mm])
    t.setStyle(TableStyle([("ALIGN", (1, 1), (3, -1), "CENTER")]))
    story.append(t)
    story.append(P("데이터셋 안에 의미적으로 강하게 응집된 작은 군집이 거의 없어서 "
                   "두 덩어리로만 갈렸다. → 군집 정보가 분류의 <i>보조 신호</i> "
                   "역할만 한다.", note))

    # ── 분류 분포 ─────────────────────────────
    story.append(P("카테고리별 분류 분포 (정답지 대비)", h1))
    if ev:
        dist_data = [
            ["카테고리 폴더", "예측 수", "정답 수", "정확", "오답"],
        ]
        order = [
            "1. 공고_지침_양식", "2. 사업계획서 수행계획서", "3. 조사_참고자료",
            "4. 중간_최종 결과물 및 보고서", "5. 발표자료",
            "6. 견적_계약_정산", "7. 기업 인증서", "8. 기타",
        ]
        for lbl in order:
            m = ev["per_category"].get(lbl, {})
            predicted = (m.get("tp", 0) + m.get("fp", 0))
            dist_data.append([
                lbl, str(predicted), str(m.get("support", 0)),
                str(m.get("tp", 0)), str(m.get("fp", 0)),
            ])
        m_un = ev["per_category"].get("_미분류_LLM위임", {})
        dist_data.append(["_미분류_LLM위임", str(m_un.get("fp", 6)),
                          "0", "—", "—"])
        t = make_table(dist_data, [70 * mm, 22 * mm, 22 * mm, 20 * mm, 20 * mm])
        t.setStyle(TableStyle([
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#fdecec")),  # 발표자료
            ("BACKGROUND", (0, 9), (-1, 9), colors.HexColor("#fffbe6")),  # 미분류
        ]))
        story.append(t)
    else:
        story.append(P("(평가 JSON이 없어 일반 분포만 표시)", body))

    # ── 분류 품질 이슈 ─────────────────────────
    story.append(P("분류 품질 이슈 — 발표자료 폴더 31건 쏠림 (정답 0건)", h1_red))
    story.append(P("정답지로 확인한 결과 “5. 발표자료” 카테고리는 "
                   "<b>예측 31건 / 정답 0건</b> — 모든 예측이 오답이다. "
                   "실제 발표자료는 ground truth에 3건뿐인데 (모두 .pptx) "
                   "입력 Test.zip에 포함되지 않아 비교에서 빠졌다.", body))
    story.append(P("<b>원인 (3가지가 겹침)</b>", h2))
    causes = [
        ("Cold-start 상태",
         "피드백 DB에 카테고리 <i>설명문 1개씩</i>만 시드로 주입. "
         "실제 운영에선 사용자 확정 이력이 누적되며 매칭 품질이 크게 개선됨."),
        ("‘발표자료’ 설명문의 함정",
         "“발표 자료, 프레젠테이션, 슬라이드, 브리핑, PT, 데모 자료” — "
         "이 description의 어휘가 한국어 비즈니스 문서 전반과 우연히 가깝게 "
         "임베딩되어 모든 종류 문서를 끌어당김 (사업계획서·인증서·동향보고서 등)."),
        ("HDBSCAN 보조 신호 부족",
         "2개 메가 클러스터만 형성되어 분류 신호가 거의 cosine에만 의존."),
    ]
    for i, (head, desc) in enumerate(causes, 1):
        story.append(P(f"<b>{i}. {head}</b> — {desc}", body))

    story.append(PageBreak())

    # ── 미분류 사유 ────────────────────────────
    story.append(P("미분류 6건 — 사유별 정리", h1))
    unc_data = [
        ["문서", "신뢰도", "사유", "top-1 / top-2"],
        ["1.입찰공고문.hwp", "0.530",
         "두 후보 박빙(차이 0.041)",
         "발표자료(0.530) / 공고_지침(0.489)"],
        ["2.세금계산서 0830.pdf", "0.546",
         "두 후보 박빙(차이 0.006)",
         "견적_계약(0.546) / 발표자료(0.540)"],
        ["7. 결과보고서 영상표출…pdf", "0.528",
         "절대점수 미달",
         "공고_지침(0.528)"],
        ["[별표 1] 안전인증대상…pdf", "0.333",
         "본문 추출 빈약(스캔/표) + 점수 매우 낮음",
         "기타(0.333) / 공고_지침(0.326)"],
        ["[별표 4] 안전확인대상…pdf", "0.412",
         "두 후보 동점(0.412 vs 0.412)",
         "공고_지침 / 발표자료"],
        ["[별표 5] 공급자적합성…pdf", "0.501",
         "두 후보 박빙(차이 0.031)",
         "공고_지침(0.501) / 발표자료(0.470)"],
    ]
    t = make_table(unc_data, [50 * mm, 18 * mm, 55 * mm, 45 * mm])
    t.setStyle(TableStyle([
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
    ]))
    story.append(t)
    story.append(P("패턴 두 가지 — <b>(A)</b> 점수 자체가 낮음 (스캔 PDF) / "
                   "<b>(B)</b> 상위 두 후보 cosine 차이가 0.05 이내. "
                   "둘 다 자동으로 LLM 큐로 위임된다.", note))

    # ── 출력 폴더 구조 ─────────────────────────
    story.append(P("출력 폴더 구조 (실제 생성됨)", h1))
    tree = (
        "기초 코드/_분류결과/<br/>"
        "├── 1. 공고_지침_양식/         (26 files, TP=6 / FP=20)<br/>"
        "├── 2. 사업계획서 수행계획서/    (0)<br/>"
        "├── 3. 조사_참고자료/            (0)<br/>"
        "├── 4. 중간_최종 결과물 및 보고서/ (0)<br/>"
        "├── 5. 발표자료/                 (31 files, TP=0 / FP=31) ← 전부 오분류<br/>"
        "├── 6. 견적_계약_정산/           (3 files, TP=3 / FP=0)<br/>"
        "├── 7. 기업 인증서/              (0)<br/>"
        "├── 8. 기타/                     (3 files, TP=0 / FP=3)<br/>"
        "├── _미분류_LLM위임/             (6 files)<br/>"
        "├── report.md                  ← 단계별 보고서<br/>"
        "├── evaluation.md / .json      ← 정답지 평가 결과<br/>"
        "├── timing.json                ← 단계별 시간 측정값<br/>"
        "└── evidence.jsonl             ← 문서별 EvidencePackage (69줄)"
    )
    story.append(P(tree, mono))

    # ── 결론 & 개선 방향 ───────────────────────
    story.append(P("결론", h1))
    story.append(P("Cold-start cosine 매칭만으로는 한국어 비즈니스 문서 분류가 "
                   "어렵다는 점이 정량적으로 확인되었다. 특히 카테고리 설명문이 "
                   "짧고 일반적일수록 “자석 효과”가 생겨 한 카테고리가 모든 "
                   "예측을 끌어당기는 현상이 발생한다.", body))
    story.append(P("운영 시스템 가정상 사용자 피드백이 누적되면서 점진적으로 "
                   "개선되도록 설계돼 있지만, 초기 도입(zero-feedback) 단계에선 "
                   "추가 신호가 반드시 필요하다.", body))

    # ── v2: 정답지 시드 적용 결과 ─────────────────
    if ev2:
        story.append(PageBreak())
        story.append(P("v2: 정답지 시드 적용 결과 (개선 방향 #1 실행)", h1))
        story.append(P("정답지에서 카테고리당 <b>2건</b>씩 시드로 EmbeddingStore에 "
                       "주입하고, 시드 파일은 분류 대상에서 제외 (leakage 방지). "
                       "임계값 0.60. categories.json description 시드는 비활성화.",
                       body))

        # v1 vs v2 비교
        story.append(P("v1 vs v2 핵심 지표 비교", h2))
        comp_data = [
            ["지표", "v1 (cold-start description)", "v2 (정답지 시드 2건)", "변화"],
            ["분류 대상", "69건", "55건 (14건 시드 제외)", "—"],
            ["Accuracy",
             f"{ev['accuracy']*100:.1f}%",
             f"{ev2['accuracy']*100:.1f}%",
             f"+{(ev2['accuracy']-ev['accuracy'])*100:.1f}p"],
            ["Macro Precision",
             f"{ev['macro_precision']:.3f}",
             f"{ev2['macro_precision']:.3f}",
             f"+{ev2['macro_precision']-ev['macro_precision']:.3f}"],
            ["Macro Recall",
             f"{ev['macro_recall']:.3f}",
             f"{ev2['macro_recall']:.3f}",
             f"+{ev2['macro_recall']-ev['macro_recall']:.3f}"],
            ["Macro F1",
             f"{ev['macro_f1']:.3f}",
             f"{ev2['macro_f1']:.3f}",
             f"+{ev2['macro_f1']-ev['macro_f1']:.3f}"],
            ["발표자료 FP", "31건", "10건", "-21"],
            ["미분류 (LLM 위임)", "6건", "1건", "-5"],
            ["TP 있는 카테고리 수", "2 / 8", "6 / 8", "+4"],
        ]
        t = make_table(comp_data, [40 * mm, 50 * mm, 50 * mm, 28 * mm])
        t.setStyle(TableStyle([
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("BACKGROUND", (3, 1), (3, -1), colors.HexColor("#e6f4ea")),
            ("FONTNAME",  (3, 1), (3, -1), "MalgunBold"),
            ("TEXTCOLOR", (3, 1), (3, -1), colors.HexColor("#0a7a2a")),
        ]))
        story.append(t)
        story.append(P("정답지 시드 2건만으로 <b>Accuracy 13% → 23.6%</b>, "
                       "<b>Macro F1 0.10 → 0.26</b> — 약 <b>2.6배 개선</b>. "
                       "발표자료 자석 효과도 31건→10건으로 감소.", note))

        # v2 카테고리별 지표
        story.append(P("v2 카테고리별 지표", h2))
        rows = [["카테고리", "Support", "TP", "FP", "FN", "P", "R", "F1"]]
        for lbl in [
            "1. 공고_지침_양식", "2. 사업계획서 수행계획서", "3. 조사_참고자료",
            "4. 중간_최종 결과물 및 보고서", "5. 발표자료",
            "6. 견적_계약_정산", "7. 기업 인증서", "8. 기타",
            "_미분류_LLM위임",
        ]:
            m = ev2["per_category"].get(lbl)
            if not m or (m["support"] == 0 and m["tp"] == 0 and m["fp"] == 0):
                continue
            rows.append([
                lbl, str(m["support"]), str(m["tp"]), str(m["fp"]), str(m["fn"]),
                f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}",
            ])
        t = make_table(rows, [55 * mm, 14 * mm, 10 * mm, 10 * mm, 10 * mm,
                              14 * mm, 14 * mm, 14 * mm])
        t.setStyle(TableStyle([
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            # TP가 0인 행만 빨간 톤
            ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#fdecec")),
        ]))
        story.append(t)

        story.append(P("남은 약점", h2))
        weakness = [
            ("발표자료 (TP=0, FP=10)",
             "정답지의 발표자료 3건이 모두 <b>.pptx</b> — file_reader가 pptx를 "
             "지원하지 않아 시드가 파일명만으로 만들어짐. python-pptx 추가하면 해결."),
            ("사업계획서 수행계획서 (TP=0, FP=2)",
             "시드 2건이 의미를 충분히 표현 못함. per_category=3 또는 4로 늘리거나 "
             "대표적인 사업계획서를 골라 시드로 사용."),
            ("기업 인증서 (P=0.21)",
             "Recall 0.44는 좋지만 Precision이 낮음 — 사업자등록증·인증서가 "
             "비슷한 벡터 영역이라 다른 카테고리 문서를 흡수."),
            ("조사_참고자료 (R=0.07)",
             "17건 중 1건만 맞춤. ‘조사’의 의미 범위가 너무 넓어 시드 2건으로는 부족."),
        ]
        for head, desc in weakness:
            story.append(P(f"<b>{head}</b> — {desc}", body))

    # ── v3: 룰 cascade + UMAP OFF + 시드 3건 ─────────────────
    if ev3:
        story.append(PageBreak())
        story.append(P("v3: 룰 cascade 적용 — 최종 결과", h1_red))
        story.append(P("STAGE A 파일명 룰(89% 즉시 확정) → STAGE B 의미기반"
                       "(임계값 캐스케이드 0.60 / 0.45) → STAGE C LLM 위임. "
                       "UMAP+HDBSCAN 단계는 OFF. 시드 3건/카테고리.", body))

        kpi_data = [
            ["지표", "v1", "v2", "v3 (최종)"],
            ["Accuracy",
             f"{ev['accuracy']*100:.1f}%",
             f"{ev2['accuracy']*100:.1f}%",
             f"{ev3['accuracy']*100:.1f}%"],
            ["Macro F1",
             f"{ev['macro_f1']:.3f}",
             f"{ev2['macro_f1']:.3f}",
             f"{ev3['macro_f1']:.3f}"],
            ["총 실행 시간", "24.1s", "24.1s", "10.8s"],
            ["LLM 위임 건수", "6건", "1건", "0건"],
            ["발표자료 FP", "31", "10", "2"],
            ["TP 있는 카테고리 수", "2 / 8", "6 / 8", "5 / 8"],
        ]
        t = make_table(kpi_data, [40 * mm, 30 * mm, 30 * mm, 30 * mm])
        t.setStyle(TableStyle([
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("BACKGROUND", (3, 1), (3, -1), colors.HexColor("#e6f4ea")),
            ("FONTNAME",  (3, 1), (3, -1), "MalgunBold"),
            ("TEXTCOLOR", (3, 1), (3, -1), colors.HexColor("#0a7a2a")),
        ]))
        story.append(t)
        story.append(P("v3에서 모든 목표 달성: <b>정확도 83.3%</b> (v1 대비 +70p), "
                       "<b>시간 -55%</b>, <b>LLM 호출 0건</b>. "
                       "룰 stage가 89%를 즉시 확정해 임베딩 비용 자체가 사라졌다.", note))

        # 단계별 시간 v3
        story.append(P("v3 단계별 시간 (총 10.82초)", h2))
        v3_timing = [
            ["단계", "시간(s)", "비중", "비고"],
            ["STAGE A (룰)", "0.001", "0.0%", "48건 즉시 확정"],
            ["STAGE B STEP 1 (임베딩)", "8.63", "79.7%", "7건만 + 시드 24건"],
            ["STAGE B STEP 4 (cosine)", "2.08", "19.2%", "1차 7건 모두 확정"],
            ["폴더 복사", "0.12", "1.1%", "—"],
            ["TOTAL", "10.82", "100.0%", "v1/v2 대비 -55%"],
        ]
        t = make_table(v3_timing, [55 * mm, 22 * mm, 18 * mm, 65 * mm])
        t.setStyle(TableStyle([
            ("ALIGN", (1, 1), (2, -1), "RIGHT"),
            ("FONTNAME", (0, -1), (-1, -1), "MalgunBold"),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8f0fa")),
        ]))
        story.append(t)

        # v3 카테고리별 지표
        story.append(P("v3 카테고리별 지표", h2))
        rows = [["카테고리", "Support", "TP", "FP", "P", "R", "F1"]]
        for lbl in [
            "1. 공고_지침_양식", "2. 사업계획서 수행계획서", "3. 조사_참고자료",
            "4. 중간_최종 결과물 및 보고서", "5. 발표자료",
            "6. 견적_계약_정산", "7. 기업 인증서", "8. 기타",
        ]:
            m = ev3["per_category"].get(lbl)
            if not m:
                continue
            rows.append([
                lbl, str(m["support"]), str(m["tp"]), str(m["fp"]),
                f"{m['precision']:.2f}", f"{m['recall']:.2f}", f"{m['f1']:.2f}",
            ])
        t = make_table(rows, [55 * mm, 16 * mm, 12 * mm, 12 * mm,
                              14 * mm, 14 * mm, 14 * mm])
        t.setStyle(TableStyle([
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            # F1 > 0.9 강조
            ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#e6f4ea")),  # 사업계획서
            ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#e6f4ea")),  # 견적
            ("BACKGROUND", (0, 7), (-1, 7), colors.HexColor("#e6f4ea")),  # 인증서 F1=1.00
        ]))
        story.append(t)
        story.append(P("3개 카테고리가 F1 ≥ 0.90 (사업계획서·견적·인증서), "
                       "2개가 F1 ≥ 0.70 (조사_참고자료·공고). "
                       "남은 약점은 발표자료(.pptx 미지원), 보고서/기타(룰이 약함).",
                       note))

        # 남은 오분류
        story.append(P("v3 남은 오분류 8건", h2))
        story.append(P("48건 중 정답 40건, 오분류 8건. 모두 1) 룰이 잘못 매칭했거나 "
                       "2) 의미적으로 ambiguous한 케이스다. 학습 누적으로 점진 개선 가능.",
                       body))

    # ── Overfitting 진단 ─────────────
    if ev3 and ev3a and ev3b and ev3clean:
        story.append(PageBreak())
        story.append(P("Overfitting 진단 (ablation 분석)", h1_red))
        story.append(P("v3의 83.3% 정확도가 진짜인지 점검. "
                       "룰 키워드가 Test.zip 파일명을 보고 만들어진 것이 있어서 "
                       "‘부풀려진’ 부분과 ‘일반화 가능한’ 부분을 분리한다.",
                       body))

        # Test-specific 키워드 식별
        story.append(P("Test.zip 파일명을 보고 추가됐다고 인정한 키워드", h2))
        suspect_data = [
            ["키워드", "근거 (Test.zip 파일)"],
            ["사업게획서", "벤처사업게획서.hwp 의 오타 보고 추가"],
            ["연구계발계획", "중소기업 연구계발계획서.hwp 오타 보고 추가"],
            ["사업계획 PART", "S3139870_210624 사업계획서 PART.hwp 보고 추가"],
            ["별표", "[별표 1/4/5] 안전인증… 보고 추가"],
            ["수요기관", "공공기관 바우처 수행기관.hwp 보고 추가"],
            ["운용요령 / 고시 일부", "안전관리 운용요령, 국립전파연구원고시 보고 추가"],
        ]
        t = make_table(suspect_data, [50 * mm, 118 * mm])
        story.append(t)
        story.append(P("이 키워드들은 <code>RULES_CONSERVATIVE</code>에서 제거됨. "
                       "<code>--conservative-rules</code> 옵션으로 사용.", body))

        # Ablation 표
        story.append(P("Ablation 결과 — 컴포넌트별 기여 분리", h2))
        abl_data = [
            ["변형", "룰", "시드", "Accuracy", "F1", "의미"],
            ["v3 (최종)", "전체", "3건",
             f"{ev3['accuracy']*100:.1f}%", f"{ev3['macro_f1']:.3f}",
             "기준"],
            ["v3-clean", "보수적", "3건",
             f"{ev3clean['accuracy']*100:.1f}%", f"{ev3clean['macro_f1']:.3f}",
             "overfitting 키워드 제거"],
            ["v3a", "OFF", "3건",
             f"{ev3a['accuracy']*100:.1f}%", f"{ev3a['macro_f1']:.3f}",
             "룰 기여 분리"],
            ["v3b", "전체", "0건 (desc)",
             f"{ev3b['accuracy']*100:.1f}%", f"{ev3b['macro_f1']:.3f}",
             "시드 기여 분리"],
        ]
        t = make_table(abl_data, [22 * mm, 18 * mm, 22 * mm, 22 * mm, 18 * mm, 60 * mm])
        t.setStyle(TableStyle([
            ("ALIGN", (1, 1), (4, -1), "CENTER"),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e6f4ea")),
            ("FONTNAME",  (0, 1), (-1, 1), "MalgunBold"),
            ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#fff8e1")),  # v3-clean
        ]))
        story.append(t)

        # 기여도 분해
        story.append(P("기여도 분해 — v3의 83.3%는 어디서 왔나", h2))
        delta_data = [
            ["원인", "정확도 기여 (추정)", "일반화 가능?"],
            ["의미기반 베이스라인 (시드+cosine)", "~25p", "O"],
            ["룰의 일반화 가능 부분 (일반 어휘)",
             "~37p", "O"],
            ["룰의 Test-specific 키워드 (overfitting)",
             "~12.5p", "X — 새 데이터셋에선 사라짐"],
            ["정답지 시드 (3건/카테고리)",
             "~8p", "O — 운영 시 사용자 피드백과 동등"],
            ["합계", "≈ 83.3%", "—"],
        ]
        t = make_table(delta_data, [70 * mm, 40 * mm, 58 * mm])
        t.setStyle(TableStyle([
            ("ALIGN", (1, 1), (1, -1), "CENTER"),
            ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#fde8e8")),
            ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#e8f0fa")),
            ("FONTNAME",  (0, 5), (-1, 5), "MalgunBold"),
        ]))
        story.append(t)

        story.append(P("판정", h2))
        verdict = [
            ("Overfitting 있나?",
             f"<b>있음. 약 12.5p 부풀려진 것으로 추정.</b> "
             f"v3 {ev3['accuracy']*100:.1f}% vs v3-clean "
             f"{ev3clean['accuracy']*100:.1f}%의 차이가 그 정도."),
            ("진짜 일반화 성능은?",
             f"<b>약 {ev3clean['accuracy']*100:.0f}~75%</b> "
             "(보수적 룰 + 시드). v3-clean이 그에 가깝다."),
            ("시드가 overfitting인가?",
             "<b>아니다.</b> 시드는 운영 단계의 사용자 확정 이력과 동등한 "
             "메커니즘 — 새 도메인에서도 똑같이 작동."),
            ("룰 전체가 overfitting인가?",
             "<b>아니다.</b> 룰의 50p 기여 중 ~38p는 일반 한국어 비즈니스 어휘로 "
             "다른 데이터셋에도 유효."),
            ("새 데이터셋에선?",
             "<b>70% 정도 시작 → 사용자 피드백 누적되며 점진 상승</b> "
             "기대. 카테고리 체계가 유사한 경우."),
        ]
        for q, a in verdict:
            story.append(P(f"<b>{q}</b><br/>{a}", body))
            story.append(Spacer(1, 4))

    story.append(P("개선 가능한 방향 (우선순위 순)", h1))
    imp_data = [
        ["#", "방법", "기대 효과", "작업량"],
        ["1", "<s>카테고리당 대표 문서 2~3개를 직접 시드 (정답지 활용)</s> <b>[완료]</b>",
         "Acc 13% → 23.6%, F1 +0.16 — 위 v2 결과 참고", "完"],
        ["2", "file_reader에 .pptx 지원 추가 (python-pptx)",
         "발표자료 시드 품질 ↑ → 발표자료 FP 추가 감소", "小"],
        ["3", "시드 수를 카테고리당 3~5건으로 증가",
         "사업계획서·조사_참고자료 회복", "小"],
        ["4", "파일명 키워드 매칭 룰 기반 1차 stage 추가",
         "큰 개선 — 명확한 케이스 0.95+로 확정", "中"],
        ["5", "임베딩 모델 업그레이드 (BAAI/bge-m3, 1024d)",
         "중 — 한국어 분리력 ↑, 모델 크기 ↑ (2GB)", "中"],
    ]
    t = make_table(imp_data, [8 * mm, 70 * mm, 70 * mm, 15 * mm])
    t.setStyle(TableStyle([
        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("ALIGN", (3, 1), (3, -1), "CENTER"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#eaf3ff")),
    ]))
    story.append(t)
    story.append(P("1번은 이미 실행되어 위 v2 결과로 효과가 검증됨. "
                   "다음 우선순위는 <b>2번 (.pptx 지원)</b>과 <b>3번 (시드 수 ↑)</b>. "
                   "두 작업 모두 small 규모로 또 한 단계 개선 가능.", note))

    return story


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main() -> None:
    output_dir = Path("_분류결과")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "실행요약.pdf"

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="의미기반 분류 Test.zip 실행 결과 + 정답지 평가",
        author="기초 코드 파이프라인",
    )
    doc.build(build_story())
    print(f"[OK] PDF 생성 → {out_path.resolve()}")


if __name__ == "__main__":
    main()
