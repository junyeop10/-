"""Enterprise document pattern knowledge used for synthetic training and evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentPatternProfile:
    type: str
    aliases: tuple[str, ...]
    profile_text: str
    tags: tuple[str, ...]
    semantic_signals: tuple[str, ...]
    layout_signals: tuple[str, ...]
    structural_signals: tuple[str, ...]
    ocr_signals: tuple[str, ...]
    numeric_patterns: tuple[str, ...]
    document_examples: tuple[str, ...]
    business_use_cases: tuple[str, ...]
    core_features: tuple[str, ...]

    def to_profile_signals(self) -> dict[str, Any]:
        return {
            "aliases": list(self.aliases),
            "semantic_signals": list(self.semantic_signals),
            "layout_signals": list(self.layout_signals),
            "structural_signals": list(self.structural_signals),
            "ocr_signals": list(self.ocr_signals),
            "numeric_patterns": list(self.numeric_patterns),
            "document_examples": list(self.document_examples),
            "business_use_cases": list(self.business_use_cases),
            "core_features": list(self.core_features),
        }


DOCUMENT_PATTERN_PROFILES: tuple[DocumentPatternProfile, ...] = (
    DocumentPatternProfile(
        type="계약서",
        aliases=("계약서", "Contract", "Agreement"),
        profile_text=(
            "계약서는 당사자 간 권리와 의무를 정의하는 문서다. 제1조, 제2조 같은 조항 구조를 가지며 "
            "갑/을 표현, 계약기간, 손해배상, 비밀유지, 준거법, 계약 해지 표현이 자주 등장한다. "
            "문단 중심 구조이며 긴 문장과 법률 표현이 많고 서명 영역이 포함될 수 있다."
        ),
        tags=("법률", "계약", "조항"),
        semantic_signals=("제1조", "제2조", "갑", "을", "계약기간", "손해배상", "비밀유지", "준거법", "계약 해지"),
        layout_signals=("dense text", "긴 문단 반복", "조항 block 반복", "서명 영역"),
        structural_signals=("조항 기반 구조", "제1조~제N조 반복", "긴 문장 비율 높음"),
        ocr_signals=("계약서", "조", "갑", "을", "체결", "서명"),
        numeric_patterns=("날짜", "기간", "조항 번호"),
        document_examples=("근로계약서", "비밀유지계약서", "서비스 이용계약서"),
        business_use_cases=("법무 검토", "계약 관리", "문서 보관"),
        core_features=("clause_pattern_score", "legal_term_density", "long_sentence_ratio", "dense_text_score", "signature_area_score"),
    ),
    DocumentPatternProfile(
        type="영수증",
        aliases=("영수증/매출전표", "Receipt", "Sales Slip"),
        profile_text=(
            "영수증 또는 매출전표는 결제 사실과 금액 정보를 기록하는 문서다. 승인번호, 카드번호, 결제일시, "
            "합계, 공급가액, 부가세, 가맹점 표현이 자주 등장한다. 숫자와 금액 비율이 높고 짧은 줄이 반복된다."
        ),
        tags=("결제", "금액", "영수증"),
        semantic_signals=("합계", "부가세", "승인번호", "카드번호", "결제일시", "공급가액", "가맹점"),
        layout_signals=("세로형", "짧은 line 반복", "숫자 밀집", "금액 정렬"),
        structural_signals=("짧은 줄 반복", "금액 row 반복"),
        ocr_signals=("TOTAL", "VAT", "승인", "합계", "카드", "금액"),
        numeric_patterns=("numeric_ratio 높음", "price column 반복", "금액 쉼표 패턴"),
        document_examples=("카드 영수증", "매출전표", "간이 영수증"),
        business_use_cases=("경비 처리", "회계 증빙", "결제 확인"),
        core_features=("receipt_pattern_score", "numeric_density", "currency_density", "repeated_price_pattern", "narrow_width_score"),
    ),
    DocumentPatternProfile(
        type="세금계산서",
        aliases=("세금계산서/Invoice", "Invoice", "Tax Invoice"),
        profile_text=(
            "세금계산서 또는 인보이스는 거래 품목과 금액 정보를 기록하는 문서다. 공급자, 공급받는자, "
            "사업자등록번호, 세액, 합계금액, 품목, 단가, 수량 표현이 자주 등장한다. 표 구조와 금액 열이 반복된다."
        ),
        tags=("세금", "계산서", "거래"),
        semantic_signals=("공급자", "공급받는자", "사업자등록번호", "세액", "합계금액", "품목", "단가", "수량"),
        layout_signals=("header metadata block", "table 중심", "품목 row 반복"),
        structural_signals=("공급자/공급받는자 영역", "품목 table", "금액 열"),
        ocr_signals=("Invoice", "사업자등록번호", "Tax", "Qty", "Amount"),
        numeric_patterns=("사업자등록번호", "numeric column 반복", "단가/수량/금액"),
        document_examples=("전자세금계산서", "인보이스", "거래명세서"),
        business_use_cases=("매입매출 관리", "세무 증빙", "ERP 입력"),
        core_features=("invoice_pattern_score", "business_id_pattern", "table_structure_score", "numeric_column_score"),
    ),
    DocumentPatternProfile(
        type="구매발주서",
        aliases=("구매발주서(PO)", "Purchase Order", "PO"),
        profile_text=(
            "구매발주서는 구매 요청과 납품 조건을 기록하는 문서다. Purchase Order, PO Number, Vendor, Buyer, "
            "Delivery Date, Payment Terms 표현이 자주 등장하며 수량과 단가가 포함된 표가 많다."
        ),
        tags=("구매", "발주", "PO"),
        semantic_signals=("Purchase Order", "PO Number", "Vendor", "Buyer", "Delivery Date", "Payment Terms"),
        layout_signals=("승인 block", "header metadata 많음", "표 기반"),
        structural_signals=("vendor/buyer 구분", "approval block", "line item table"),
        ocr_signals=("PO", "Vendor", "Qty", "Unit Price", "Delivery"),
        numeric_patterns=("수량", "단가", "납기일", "PO 번호"),
        document_examples=("구매발주서", "Purchase Order", "납품 발주서"),
        business_use_cases=("구매 관리", "납품 추적", "ERP 발주 입력"),
        core_features=("po_number_pattern", "approval_block_score", "vendor_block_score", "table_density"),
    ),
    DocumentPatternProfile(
        type="발표자료",
        aliases=("발표자료(Presentation)", "Presentation", "Slides", "PPT"),
        profile_text=(
            "발표자료는 핵심 내용을 요약하여 전달하는 문서다. 큰 제목과 짧은 bullet 문장이 많고 이미지와 도표 비율이 높다. "
            "문장 길이가 짧고 페이지당 텍스트 양이 적은 경우가 많다."
        ),
        tags=("슬라이드", "발표", "요약"),
        semantic_signals=("Overview", "Agenda", "Summary", "Conclusion", "Next Step"),
        layout_signals=("큰 제목", "bullet 많음", "이미지 많음", "페이지당 텍스트 적음"),
        structural_signals=("slide 기반 반복", "large header 반복", "짧은 bullet"),
        ocr_signals=("Agenda", "Summary", "Title"),
        numeric_patterns=("숫자 비율 낮음", "표 적음"),
        document_examples=("최종발표자료", "교육자료", "사업 제안 발표"),
        business_use_cases=("회의 발표", "교육", "제안 설명"),
        core_features=("slide_like_layout_score", "bullet_density", "large_header_score", "image_area_ratio"),
    ),
    DocumentPatternProfile(
        type="논문",
        aliases=("논문/기술문서", "Paper", "Technical Document", "Research"),
        profile_text=(
            "논문 또는 기술 연구 문서는 연구 목적과 결과를 설명하는 문서다. abstract, method, results, references, DOI, "
            "citation 표현이 자주 등장한다. 텍스트 밀도가 높고 2단 구조와 참고문헌 영역이 나타날 수 있다."
        ),
        tags=("연구", "논문", "기술"),
        semantic_signals=("abstract", "method", "results", "references", "doi", "citation"),
        layout_signals=("dense text", "2-column", "figure/table caption"),
        structural_signals=("abstract-method-results-references", "참고문헌 마지막 페이지", "figure/table 번호"),
        ocr_signals=("Abstract", "References", "Fig", "Table"),
        numeric_patterns=("citation number 반복", "figure/table 번호"),
        document_examples=("review paper", "technical report", "research article"),
        business_use_cases=("R&D 조사", "기술 검토", "문헌 관리"),
        core_features=("citation_density", "two_column_score", "research_structure_score", "references_last_page_score"),
    ),
    DocumentPatternProfile(
        type="이력서",
        aliases=("이력서/CV", "Resume", "CV"),
        profile_text=(
            "이력서 또는 CV는 개인의 경력, 학력, 기술과 연락처를 정리한 문서다. Experience, Education, Skills, "
            "Email, Phone 표현이 자주 등장하며 section divider와 bullet 기반 경력 설명이 많다."
        ),
        tags=("인사", "채용", "경력"),
        semantic_signals=("경력", "학력", "기술", "Experience", "Education", "Skills"),
        layout_signals=("상단 연락처", "section divider 반복", "bullet 기반 경력 설명"),
        structural_signals=("경력/학력 section 반복", "연락처 header", "기간 반복"),
        ocr_signals=("Email", "Phone", "Experience"),
        numeric_patterns=("year/month 기간", "전화번호", "이메일"),
        document_examples=("지원자 이력서", "CV", "경력기술서"),
        business_use_cases=("채용 검토", "인재 DB", "HR 문서 관리"),
        core_features=("resume_layout_score", "contact_pattern_score", "career_section_score"),
    ),
    DocumentPatternProfile(
        type="일반 보고서",
        aliases=("보고서", "Report", "Business Report"),
        profile_text=(
            "일반 보고서는 현황, 분석, 결론, 성과를 정리하는 문서다. 요약, 현황, 분석, 결론 표현이 자주 등장하며 "
            "heading hierarchy와 표/차트가 혼합될 수 있다."
        ),
        tags=("보고", "분석", "업무"),
        semantic_signals=("요약", "현황", "분석", "결론", "성과"),
        layout_signals=("heading hierarchy", "표/차트 혼합", "본문과 도표 혼합"),
        structural_signals=("요약-분석-결론", "section heading", "표/차트 설명"),
        ocr_signals=("분석", "결론", "보고"),
        numeric_patterns=("성과 지표", "표 수치", "기간별 수치"),
        document_examples=("업무보고서", "분석보고서", "주간보고"),
        business_use_cases=("경영 보고", "성과 분석", "업무 공유"),
        core_features=("report_structure_score", "heading_density", "chart_presence_score"),
    ),
)


ADDITIONAL_DOCUMENT_PATTERN_PROFILES: tuple[DocumentPatternProfile, ...] = (
    DocumentPatternProfile(
        type="인증서",
        aliases=("증명서", "확인서", "Certificate", "Certification"),
        profile_text=(
            "인증서 또는 확인서는 특정 자격, 상태, 등록, 선정, 유효기간을 증명하는 문서다. "
            "인증번호, 확인번호, 발급기관, 유효기간, 대표자, 회사명, 직인 또는 서명 영역이 자주 등장한다."
        ),
        tags=("인증", "증명", "기업"),
        semantic_signals=("인증서", "확인서", "증명서", "인증번호", "확인번호", "발급기관", "유효기간", "대표자"),
        layout_signals=("중앙 제목", "넓은 여백", "하단 직인 영역", "서명 영역"),
        structural_signals=("상단 기관명", "중앙 제목", "본문 증명 문장", "하단 발급일/기관"),
        ocr_signals=("인증", "확인", "증명", "발급", "유효기간", "직인"),
        numeric_patterns=("인증번호", "확인번호", "발급일", "유효기간"),
        document_examples=("벤처기업인증서", "중소기업확인서", "지방세완납증명서", "기업부설연구소 인정서"),
        business_use_cases=("입찰 제출", "정부지원사업", "거래처 증빙", "기업 자격 확인"),
        core_features=("certificate_pattern_score", "centered_title_score", "signature_area_score", "whitespace_balance_score"),
    ),
    DocumentPatternProfile(
        type="사업자등록증",
        aliases=("사업자 등록증", "Business Registration", "Registration Certificate"),
        profile_text=(
            "사업자등록증은 사업자의 등록 정보를 증명하는 문서다. 사업자등록번호, 상호, 법인명, 대표자, "
            "개업연월일, 사업장 소재지, 업태, 종목, 관할세무서 표현이 자주 등장한다."
        ),
        tags=("기업", "등록", "세무"),
        semantic_signals=("사업자등록증", "사업자등록번호", "상호", "법인명", "대표자", "개업연월일", "사업장 소재지", "업태", "종목"),
        layout_signals=("표 형태 메타데이터", "상단 제목", "항목-값 구조", "관공서 서식"),
        structural_signals=("등록번호 영역", "대표자/상호 영역", "사업장 주소 영역", "업태/종목 행"),
        ocr_signals=("사업자", "등록번호", "상호", "대표자", "개업", "세무서"),
        numeric_patterns=("사업자등록번호", "날짜", "우편번호"),
        document_examples=("사업자등록증", "법인 사업자등록증", "개인 사업자등록증"),
        business_use_cases=("거래처 등록", "세무 증빙", "계약 첨부", "입찰 서류"),
        core_features=("business_id_pattern", "table_structure_score", "header_block_score", "certificate_pattern_score"),
    ),
    DocumentPatternProfile(
        type="법인등기부등본",
        aliases=("등기사항전부증명서", "Corporate Registry", "Registry Certificate"),
        profile_text=(
            "법인등기부등본 또는 등기사항전부증명서는 법인의 등기 정보를 보여주는 문서다. 상호, 본점, 목적, "
            "임원, 대표이사, 회사성립연월일, 발행주식, 자본금, 등기기록 표현이 반복된다."
        ),
        tags=("법인", "등기", "기업"),
        semantic_signals=("법인등기부등본", "등기사항전부증명서", "상호", "본점", "목적", "임원", "대표이사", "자본금", "등기기록"),
        layout_signals=("관공서 문서", "항목-값 블록", "표와 긴 문장 혼합", "페이지 번호"),
        structural_signals=("법인 기본정보", "목적 목록", "임원 목록", "등기기록 반복"),
        ocr_signals=("등기", "법인", "본점", "대표이사", "자본금"),
        numeric_patterns=("법인등록번호", "회사성립연월일", "자본금", "주식 수"),
        document_examples=("법인등기부등본", "등기사항전부증명서", "말소사항 포함 등본"),
        business_use_cases=("법무 확인", "거래처 심사", "계약 첨부", "투자 검토"),
        core_features=("registry_pattern_score", "table_structure_score", "dense_text_score", "section_divider_score"),
    ),
    DocumentPatternProfile(
        type="재무제표",
        aliases=("재무제표증명", "Financial Statement", "Balance Sheet", "Income Statement"),
        profile_text=(
            "재무제표는 기업의 재무상태와 손익을 나타내는 문서다. 자산, 부채, 자본, 매출액, 영업이익, "
            "당기순이익, 과세기간, 손익계산서, 재무상태표 표현과 금액 열이 자주 등장한다."
        ),
        tags=("회계", "재무", "세무"),
        semantic_signals=("재무제표", "재무상태표", "손익계산서", "자산", "부채", "자본", "매출액", "영업이익", "당기순이익"),
        layout_signals=("표 중심", "숫자 열 반복", "계정과목 행", "금액 정렬"),
        structural_signals=("계정과목", "당기/전기 비교", "합계 행", "주석 영역"),
        ocr_signals=("자산", "부채", "자본", "매출", "손익", "합계"),
        numeric_patterns=("금액 열", "천단위 쉼표", "당기/전기 수치", "음수 금액"),
        document_examples=("표준재무제표증명", "재무상태표", "손익계산서", "합계잔액시산표"),
        business_use_cases=("신용평가", "투자 검토", "세무 신고", "입찰 제출"),
        core_features=("financial_statement_score", "numeric_column_score", "table_structure_score", "dense_text_score"),
    ),
    DocumentPatternProfile(
        type="견적서",
        aliases=("Quotation", "Estimate", "Quote"),
        profile_text=(
            "견적서는 제품이나 서비스 제공 전 예상 금액과 조건을 제시하는 문서다. 견적번호, 견적일자, 공급자, "
            "수신처, 품명, 규격, 수량, 단가, 공급가액, 부가세, 합계금액, 유효기간 표현이 자주 등장한다."
        ),
        tags=("영업", "견적", "금액"),
        semantic_signals=("견적서", "견적번호", "견적일자", "수신처", "공급자", "품명", "규격", "수량", "단가", "합계금액", "유효기간"),
        layout_signals=("상단 메타데이터", "품목 표", "금액 합계 영역", "하단 조건 문구"),
        structural_signals=("견적 header", "line item table", "합계/부가세 영역", "비고/조건"),
        ocr_signals=("견적", "품명", "수량", "단가", "합계", "부가세"),
        numeric_patterns=("견적번호", "수량", "단가", "금액", "부가세"),
        document_examples=("제품 견적서", "서비스 견적서", "유지보수 견적서"),
        business_use_cases=("영업 제안", "구매 검토", "계약 전 비교", "예산 산정"),
        core_features=("quotation_pattern_score", "table_structure_score", "numeric_column_score", "total_keyword_exists"),
    ),
    DocumentPatternProfile(
        type="거래명세서",
        aliases=("Transaction Statement", "Statement of Transaction", "Delivery Statement"),
        profile_text=(
            "거래명세서는 거래 품목, 수량, 단가, 금액, 공급자와 공급받는자를 정리한 문서다. 거래일자, 품목, "
            "규격, 수량, 단가, 공급가액, 세액, 합계, 인수자 표현이 반복된다."
        ),
        tags=("거래", "품목", "회계"),
        semantic_signals=("거래명세서", "거래일자", "공급자", "공급받는자", "품목", "규격", "수량", "단가", "공급가액", "세액", "합계"),
        layout_signals=("품목 표 중심", "금액 열 반복", "거래처 메타데이터", "합계 영역"),
        structural_signals=("공급자/공급받는자", "품목 table", "합계 row", "인수자/확인 영역"),
        ocr_signals=("거래", "명세", "품목", "수량", "단가", "합계"),
        numeric_patterns=("수량", "단가", "금액", "세액", "거래일자"),
        document_examples=("거래명세서", "납품명세서", "출고명세서"),
        business_use_cases=("납품 확인", "회계 증빙", "거래처 정산", "재고 확인"),
        core_features=("transaction_statement_score", "table_structure_score", "numeric_column_score", "approval_block_score"),
    ),
    DocumentPatternProfile(
        type="회의록",
        aliases=("Meeting Minutes", "Minutes", "회의 내용"),
        profile_text=(
            "회의록은 회의의 일시, 장소, 참석자, 안건, 논의 내용, 결정 사항, 후속 조치를 기록하는 문서다. "
            "회의명, 참석자, 안건, 결정사항, Action Item, 담당자, 기한 표현이 자주 등장한다."
        ),
        tags=("회의", "업무", "기록"),
        semantic_signals=("회의록", "회의명", "일시", "장소", "참석자", "안건", "논의 내용", "결정사항", "담당자", "기한"),
        layout_signals=("상단 회의 정보", "heading hierarchy", "bullet 또는 번호 목록", "표/목록 혼합"),
        structural_signals=("회의 정보 block", "안건별 section", "결정사항", "Action Item"),
        ocr_signals=("회의", "참석자", "안건", "결정", "담당자"),
        numeric_patterns=("회의 일시", "안건 번호", "기한 날짜"),
        document_examples=("주간회의록", "프로젝트 회의록", "이사회 회의록"),
        business_use_cases=("업무 공유", "결정 이력", "프로젝트 관리", "책임 추적"),
        core_features=("meeting_minutes_score", "heading_density", "bullet_density", "section_divider_score"),
    ),
    DocumentPatternProfile(
        type="공문",
        aliases=("Official Letter", "Notice", "공문서"),
        profile_text=(
            "공문은 기관이나 회사가 공식적으로 전달하는 문서다. 수신, 참조, 제목, 시행일자, 문서번호, "
            "본문, 붙임, 담당자, 직인 표현이 자주 등장한다."
        ),
        tags=("공문", "행정", "공지"),
        semantic_signals=("공문", "수신", "참조", "제목", "시행일자", "문서번호", "붙임", "담당자", "직인"),
        layout_signals=("상단 수신/참조", "공식 제목", "본문 단락", "하단 담당자/직인"),
        structural_signals=("수신/참조 header", "제목 line", "본문", "붙임 목록"),
        ocr_signals=("수신", "참조", "제목", "붙임", "담당자"),
        numeric_patterns=("문서번호", "시행일자", "전화번호"),
        document_examples=("기관 공문", "회사 공문", "지원사업 안내 공문"),
        business_use_cases=("대외 발송", "기관 제출", "행정 처리", "공지 전달"),
        core_features=("official_letter_score", "header_block_score", "footer_pattern_score", "signature_area_score"),
    ),
    DocumentPatternProfile(
        type="사업계획서",
        aliases=("Business Plan", "Project Plan", "사업 계획"),
        profile_text=(
            "사업계획서는 사업 목적, 시장 분석, 추진 전략, 실행 계획, 예산, 기대 효과를 설명하는 문서다. "
            "사업 개요, 목표 시장, 추진 전략, 일정, 예산, 매출 계획, 기대효과 표현이 자주 등장한다."
        ),
        tags=("사업", "기획", "전략"),
        semantic_signals=("사업계획서", "사업 개요", "목표 시장", "추진 전략", "실행 계획", "예산", "매출 계획", "기대효과"),
        layout_signals=("heading hierarchy", "표/차트 혼합", "긴 문단과 bullet 혼합", "목차 가능"),
        structural_signals=("개요-시장-전략-계획-예산", "section heading", "표/차트 설명"),
        ocr_signals=("사업", "계획", "시장", "전략", "예산", "효과"),
        numeric_patterns=("예산 금액", "매출 목표", "일정", "시장 규모"),
        document_examples=("정부지원사업 사업계획서", "창업 사업계획서", "신규사업 계획서"),
        business_use_cases=("투자 유치", "정부지원사업", "내부 기획", "전략 검토"),
        core_features=("business_plan_score", "report_structure_score", "heading_density", "chart_presence_score"),
    ),
    DocumentPatternProfile(
        type="과업지시서",
        aliases=("Statement of Work", "Scope of Work", "SOW", "RFP"),
        profile_text=(
            "과업지시서는 용역이나 프로젝트에서 수행해야 할 업무 범위와 산출물을 정의하는 문서다. 과업명, "
            "과업 목적, 과업 내용, 수행 일정, 산출물, 제출 기한, 평가 기준, 제안 요청 표현이 자주 등장한다."
        ),
        tags=("프로젝트", "용역", "요구사항"),
        semantic_signals=("과업지시서", "과업명", "과업 목적", "과업 내용", "수행 일정", "산출물", "제출 기한", "평가 기준", "제안 요청"),
        layout_signals=("번호 section", "긴 요구사항 문단", "표/목록 혼합", "일정표 가능"),
        structural_signals=("과업 개요", "범위", "수행 일정", "산출물", "평가 기준"),
        ocr_signals=("과업", "용역", "산출물", "제출", "평가"),
        numeric_patterns=("일정 날짜", "예산 금액", "평가 배점", "항목 번호"),
        document_examples=("용역 과업지시서", "제안요청서", "프로젝트 SOW"),
        business_use_cases=("외주 발주", "프로젝트 범위 정의", "제안 평가", "계약 첨부"),
        core_features=("sow_pattern_score", "heading_density", "table_structure_score", "long_sentence_ratio"),
    ),
)


DATASET_DOCUMENT_PATTERN_PROFILES: tuple[DocumentPatternProfile, ...] = (
    DocumentPatternProfile(
        type="공고문",
        aliases=("모집공고", "지원사업 공고", "Announcement", "Notice"),
        profile_text=(
            "공고문은 정부기관, 공공기관, 지원사업 운영기관이 사업 참여자나 기업을 모집하기 위해 발행하는 문서다. "
            "사업명, 모집공고, 사업목적, 지원규모, 신청기간, 지원대상, 지원내용, 신청방법, 선정절차, 문의처 같은 표현이 자주 등장한다. "
            "문서 상단에는 기관 공고번호와 공고 제목이 있고, 본문은 사업개요, 지원대상, 신청기간, 제출서류, 평가절차 순서로 구성되는 경우가 많다."
        ),
        tags=("공공사업", "모집", "지원사업"),
        semantic_signals=("공고", "모집공고", "사업목적", "지원규모", "신청기간", "지원대상", "지원내용", "신청방법", "선정절차", "제출서류", "문의처", "중소기업", "지원사업"),
        layout_signals=("첫 페이지 상단 공고번호 또는 기관명", "긴 본문형 문서", "번호형 목차와 항목 반복", "표와 bullet 혼합"),
        structural_signals=("사업개요 → 지원대상 → 신청기간 → 접수방법 → 평가/선정", "공고번호 존재 가능", "기관장 명의 존재 가능"),
        ocr_signals=("공고", "사업개요", "지원대상", "신청", "접수", "선정", "기관장"),
        numeric_patterns=("공고번호", "신청기간", "지원규모", "접수 마감일"),
        document_examples=("지원사업 모집공고", "중소기업 지원사업 공고", "참여기업 모집공고"),
        business_use_cases=("지원사업 탐색", "사업 참여 검토", "제출 일정 확인"),
        core_features=("announcement_pattern_score", "support_program_terms_count", "application_period_pattern", "eligibility_section_score"),
    ),
    DocumentPatternProfile(
        type="공모안내서",
        aliases=("공모 안내서", "지원사업 안내서", "Application Guide", "Program Guide"),
        profile_text=(
            "공모안내서는 지원사업이나 프로그램의 신청 절차와 유의사항을 상세히 안내하는 문서다. "
            "공모기간, 유의사항, 신청접수, 사업관리시스템, 제출서류, 지원절차, 평가방법, 선정기준, 추진일정 같은 표현이 자주 등장한다. "
            "공고문보다 안내 성격이 강하고, 유의사항과 절차 설명, 일정표, 제출서류 목록이 자세하게 포함된다."
        ),
        tags=("공공사업", "안내", "신청절차"),
        semantic_signals=("공모안내서", "유의사항", "공모기간", "신청접수", "사업관리시스템", "제출서류", "지원절차", "평가방법", "선정기준", "추진일정", "고객센터", "문의하기"),
        layout_signals=("앞부분에 유의사항 목록", "절차형 설명 많음", "표/일정/목록 구조 포함"),
        structural_signals=("유의사항 → 신청방법 → 제출서류 → 평가절차 → 문의처", "접수 마감일/시스템 안내 반복"),
        ocr_signals=("공모", "안내서", "유의사항", "접수", "시스템", "제출"),
        numeric_patterns=("공모기간", "접수 마감일", "추진일정", "문의 전화번호"),
        document_examples=("지원사업 공모안내서", "참여기업 모집 안내서", "사업관리시스템 신청 안내"),
        business_use_cases=("신청 준비", "제출서류 확인", "사업 일정 관리"),
        core_features=("guide_document_score", "instruction_section_density", "schedule_table_score", "submission_document_terms_count"),
    ),
    DocumentPatternProfile(
        type="제안요청서",
        aliases=("제안요청서/RFP", "RFP", "Request for Proposal", "제안 요청서"),
        profile_text=(
            "제안요청서는 발주기관이 외부 업체에게 제안서 제출을 요청하기 위해 작성하는 문서다. "
            "제안요청서, 사업개요, 사업범위, 과업내용, 제안서 작성요령, 평가항목, 평가기준, 제출서류, 사업기간, 용역범위 같은 표현이 자주 등장한다. "
            "목차가 있고 평가배점표, 제안서 제출 방식, 수행 조건, 일정표가 포함되는 경우가 많다."
        ),
        tags=("발주", "제안", "RFP"),
        semantic_signals=("제안요청서", "제안서", "사업개요", "사업범위", "과업내용", "평가항목", "평가기준", "배점", "제출서류", "사업기간", "용역범위", "제안서 작성"),
        layout_signals=("목차 존재 가능", "표/평가표 많음", "조항형 설명", "사업 정보와 평가 정보가 구분됨"),
        structural_signals=("사업개요 → 과업범위 → 제안서 작성 → 평가기준 → 제출방법", "평가항목/배점표 존재 가능"),
        ocr_signals=("제안", "요청서", "평가", "배점", "과업", "제출"),
        numeric_patterns=("평가 배점", "사업기간", "제출 마감일", "용역 금액"),
        document_examples=("용역 제안요청서", "RFP", "제안서 작성 안내"),
        business_use_cases=("외주 발주", "제안 준비", "평가 기준 확인"),
        core_features=("rfp_pattern_score", "evaluation_table_score", "proposal_terms_count", "requirement_section_score"),
    ),
    DocumentPatternProfile(
        type="과업지시서",
        aliases=("Statement of Work", "Scope of Work", "SOW", "과업 지시서"),
        profile_text=(
            "과업지시서는 발주자가 수행기관에게 수행해야 할 업무 범위와 산출물을 지시하는 문서다. "
            "과업명, 용역명, 용역목적, 용역기간, 과업내용, 조사내용, 산출물, 수행방법, 보고서 제출, 기초자료 작성 같은 표현이 자주 등장한다. "
            "제안요청서와 유사하지만 평가기준보다 실제 수행 범위와 산출물 요구가 더 강하게 나타난다."
        ),
        tags=("프로젝트", "용역", "과업"),
        semantic_signals=("과업지시서", "과업내용", "용역명", "용역목적", "용역기간", "수행방법", "조사내용", "산출물", "보고서 제출", "기초자료", "조사 및 분석"),
        layout_signals=("번호형 항목 구조", "과업내용/수행방법 중심", "표보다 본문 설명 비중 높음"),
        structural_signals=("용역명 → 목적 → 기간 → 과업내용 → 산출물", "수행해야 할 세부 업무 목록 반복"),
        ocr_signals=("과업", "용역", "조사", "산출물", "수행", "보고서"),
        numeric_patterns=("용역기간", "제출 기한", "보고서 부수", "일정 날짜"),
        document_examples=("용역 과업지시서", "조사 과업지시서", "수행 범위 정의서"),
        business_use_cases=("용역 수행", "산출물 관리", "프로젝트 범위 확인"),
        core_features=("task_instruction_score", "deliverable_terms_count", "service_period_pattern", "work_scope_section_score"),
    ),
    DocumentPatternProfile(
        type="사업계획서",
        aliases=("Business Plan", "Project Plan", "사업 계획서"),
        profile_text=(
            "사업계획서는 창업, 기술개발, 지원사업, 투자유치 등을 위해 사업 목표와 추진전략을 설명하는 문서다. "
            "사업목표, 사업내용, 추진전략, 시장분석, 경쟁력, 기대효과, 사업화 전략, 수익모델, 예산, 추진일정, 기업현황 같은 표현이 자주 등장한다. "
            "목차와 장/절 구조가 있고 표, 그림, 일정표가 섞여 있는 경우가 많다."
        ),
        tags=("사업", "기획", "지원사업"),
        semantic_signals=("사업계획서", "사업목표", "사업내용", "추진전략", "시장분석", "경쟁력", "기대효과", "사업화", "수익모델", "예산", "추진일정", "기업현황"),
        layout_signals=("목차 존재 가능", "장/절 heading 구조", "표/도표/본문 혼합", "긴 분량"),
        structural_signals=("기업개요 → 사업목표 → 시장분석 → 추진전략 → 예산/일정 → 기대효과", "기술개발/창업지원 관련 항목 반복"),
        ocr_signals=("사업", "계획서", "추진", "전략", "시장", "기대효과"),
        numeric_patterns=("예산", "추진일정", "매출 목표", "시장 규모"),
        document_examples=("창업 사업계획서", "정부지원사업 사업계획서", "기술개발 사업계획서"),
        business_use_cases=("지원사업 제출", "투자유치", "내부 사업 검토"),
        core_features=("business_plan_score", "strategy_terms_count", "market_analysis_section_score", "expected_effect_terms_count"),
    ),
    DocumentPatternProfile(
        type="수행계획서",
        aliases=("직무수행계획서", "Execution Plan", "Action Plan"),
        profile_text=(
            "수행계획서는 특정 과업이나 직무를 어떻게 수행할지 계획을 제시하는 문서다. 수행계획서, 수행방안, 수행목표, "
            "세부 추진일정, 기대효과, 사후관리, 컨설팅 개요, 기업 요구내용, 분석, 수행방법 같은 표현이 자주 등장한다. "
            "사업계획서보다 실행 방법과 일정, 수행체계에 초점이 강하다."
        ),
        tags=("수행", "계획", "컨설팅"),
        semantic_signals=("수행계획서", "수행방안", "수행목표", "세부 추진일정", "기대효과", "사후관리", "컨설팅 개요", "요구내용", "수행방법", "추진일정"),
        layout_signals=("목차 존재 가능", "단계별 수행 항목", "일정표/추진체계 포함 가능"),
        structural_signals=("개요 → 요구내용 분석 → 수행목표 → 수행방안 → 일정 → 기대효과", "컨설팅/용역 수행 항목 반복"),
        ocr_signals=("수행", "계획서", "목표", "일정", "기대효과", "사후관리"),
        numeric_patterns=("세부 추진일정", "단계 번호", "기간", "일정표"),
        document_examples=("직무수행계획서", "컨설팅 수행계획서", "용역 수행계획서"),
        business_use_cases=("수행 전략 수립", "컨설팅 착수", "평가 제출"),
        core_features=("execution_plan_score", "action_plan_terms_count", "schedule_section_score", "consulting_terms_count"),
    ),
    DocumentPatternProfile(
        type="조사자료",
        aliases=("조사/참고자료", "참고자료", "Reference Material", "Research Material"),
        profile_text=(
            "조사자료나 참고자료는 법령, 고시, 시장동향, 품목 목록, 기준표, 매뉴얼, 벤치마킹 자료처럼 특정 의사결정이나 연구를 지원하기 위한 배경자료다. "
            "별표, 기준, 품목, 분류, 고시, 시행규칙, 매뉴얼, 동향, 전망, 비교, 벤치마킹 같은 표현이 자주 등장한다. "
            "표와 목록이 많고 분류표, 품목표, 법령 조항, 참고 링크가 포함될 수 있다."
        ),
        tags=("조사", "참고", "자료"),
        semantic_signals=("별표", "기준", "품목", "분류", "고시", "시행규칙", "매뉴얼", "동향", "전망", "비교", "벤치마킹", "참고자료"),
        layout_signals=("표/목록 비중 높음", "법령/분류표 구조", "긴 목록 반복"),
        structural_signals=("분류 → 품목 → 세부품목", "고시/법령/기준 문서 구조", "참고용 목록형 구조"),
        ocr_signals=("별표", "기준", "품목", "분류", "고시", "매뉴얼"),
        numeric_patterns=("별표 번호", "품목 코드", "조항 번호", "분류 번호"),
        document_examples=("시장동향 참고자료", "품목 분류표", "법령 고시 자료"),
        business_use_cases=("자료 조사", "기준 검토", "사업계획 참고"),
        core_features=("reference_material_score", "regulation_terms_count", "table_list_density", "classification_list_score"),
    ),
    DocumentPatternProfile(
        type="결과보고서",
        aliases=("결과보고서/최종보고서", "최종보고서", "Final Report", "Result Report"),
        profile_text=(
            "결과보고서 또는 최종보고서는 수행한 사업, 과업, 연구, 컨설팅의 결과를 정리하는 문서다. "
            "결과보고서, 최종보고서, 과업지시 내용, 수행결과, 사업내용, 성과, 추진실적, 문제점, 개선방안, 기대효과, 지원기관, 수행기관 같은 표현이 자주 등장한다. "
            "앞부분에 사업 정보와 수행기관 정보가 있고, 본문에는 수행 내용과 결과, 성과가 정리된다."
        ),
        tags=("보고", "결과", "성과"),
        semantic_signals=("결과보고서", "최종보고서", "수행결과", "추진실적", "성과", "문제점", "개선방안", "기대효과", "지원기관", "수행기관", "사업내용"),
        layout_signals=("표지/요약 페이지 존재 가능", "사업 정보 표 존재", "본문 + 표/도표 혼합"),
        structural_signals=("사업개요 → 수행내용 → 결과 → 성과 → 개선방안/기대효과", "지원기관/수행기관 metadata block"),
        ocr_signals=("결과", "보고서", "최종", "수행", "성과", "실적"),
        numeric_patterns=("추진실적", "성과 수치", "사업기간", "지원금"),
        document_examples=("최종보고서", "컨설팅 결과보고서", "사업 수행 결과보고서"),
        business_use_cases=("성과 제출", "사업 정산", "컨설팅 완료 보고"),
        core_features=("final_report_score", "result_terms_count", "performance_section_score", "project_metadata_block_score"),
    ),
    DocumentPatternProfile(
        type="계약서",
        aliases=("계약서/협약서", "협약서", "Contract", "Agreement"),
        profile_text=(
            "계약서 또는 협약서는 당사자 간 권리와 의무, 수행범위, 금액, 기간을 정하는 법적 성격의 문서다. "
            "계약서, 협약서, 갑, 을, 제1조, 목적, 계약기간, 협약기간, 계약금액, 정부지원금, 부담금, 손해배상, 비밀유지, 해지, 서명, 날인 같은 표현이 자주 등장한다. "
            "조항 번호가 반복되고 긴 문장과 법률 표현이 많으며 마지막에 서명 또는 날인 영역이 있다."
        ),
        tags=("법률", "계약", "협약"),
        semantic_signals=("계약서", "협약서", "갑", "을", "제1조", "목적", "계약기간", "협약기간", "계약금액", "정부지원금", "부담금", "손해배상", "비밀유지", "해지", "서명", "날인"),
        layout_signals=("dense text", "조항 block 반복", "서명/날인 영역", "금액/기간 block 존재 가능"),
        structural_signals=("제목 → 당사자 정의 → 제1조부터 조항 반복 → 서명 영역", "사업명/협약기간/협약금액 metadata 포함"),
        ocr_signals=("계약", "협약", "제1조", "갑", "을", "서명", "날인"),
        numeric_patterns=("계약기간", "협약기간", "계약금액", "정부지원금", "부담금"),
        document_examples=("용역계약서", "지원사업 협약서", "비밀유지계약서"),
        business_use_cases=("법무 검토", "협약 체결", "사업 수행 관리"),
        core_features=("contract_score", "clause_pattern_score", "agreement_terms_count", "signature_area_score"),
    ),
    DocumentPatternProfile(
        type="세금계산서",
        aliases=("세금계산서/Invoice", "전자세금계산서", "Invoice", "Tax Invoice"),
        profile_text=(
            "세금계산서는 전자세금계산서나 거래 세금 정보를 나타내는 문서다. 전자세금계산서, 승인번호, 공급자, 공급받는자, 등록번호, "
            "사업자등록번호, 상호, 성명, 작성일자, 공급가액, 세액, 품목, 수량, 단가, 합계 같은 표현이 자주 등장한다. "
            "공급자와 공급받는자 영역이 나뉘고 품목과 금액이 표 형태로 정리된다."
        ),
        tags=("세금", "계산서", "거래"),
        semantic_signals=("전자세금계산서", "승인번호", "공급자", "공급받는자", "등록번호", "사업자등록번호", "상호", "작성일자", "공급가액", "세액", "품목", "수량", "단가", "합계"),
        layout_signals=("공급자/공급받는자 양쪽 block", "품목 table", "금액 column 반복", "숫자 비율 높음"),
        structural_signals=("승인번호 → 공급자/공급받는자 → 작성일자/공급가액/세액 → 품목 table", "사업자등록번호 패턴 존재"),
        ocr_signals=("세금계산서", "승인번호", "공급자", "공급받는자", "공급가액", "세액"),
        numeric_patterns=("사업자등록번호", "승인번호", "공급가액", "세액", "합계"),
        document_examples=("전자세금계산서", "세금계산서 PDF", "인보이스"),
        business_use_cases=("세무 증빙", "정산", "회계 입력"),
        core_features=("tax_invoice_score", "business_id_pattern", "supplier_receiver_block_score", "tax_amount_pattern"),
    ),
    DocumentPatternProfile(
        type="기업증명서",
        aliases=("기업증명서/인증서", "인증서", "증명서", "확인서", "Certificate"),
        profile_text=(
            "기업증명서 또는 인증서는 사업자등록증, 벤처기업확인서, 중소기업확인서, 법인등기부등본, 납세증명, 표준재무제표증명처럼 기업의 법적 상태나 자격을 증명하는 문서다. "
            "발급번호, 사업자등록번호, 법인등록번호, 상호, 대표자, 사업장, 유효기간, 용도, 발급일, 세무서장, 확인서, 증명원, 등록증 같은 표현이 자주 등장한다. "
            "공식 양식이며 여백이 많고 발급기관, 번호, 바코드, 직인 영역이 포함될 수 있다."
        ),
        tags=("기업", "증명", "인증"),
        semantic_signals=("사업자등록증", "법인등기부등본", "벤처기업확인서", "중소기업확인서", "부가가치세과세표준증명", "표준재무제표증명", "발급번호", "사업자등록번호", "법인등록번호", "대표자", "상호", "유효기간", "용도", "세무서장"),
        layout_signals=("공식 양식", "중앙 제목", "여백 많음", "바코드/직인/기관명 영역 존재 가능", "표준화된 field-value 구조"),
        structural_signals=("발급번호 → 기업명/대표자/등록번호 → 주소/업태/종목 → 발급기관/발급일", "증명서/확인서/등록증 제목 존재"),
        ocr_signals=("증명", "확인서", "등록증", "발급번호", "사업자등록번호", "대표자", "유효기간"),
        numeric_patterns=("발급번호", "사업자등록번호", "법인등록번호", "발급일", "유효기간"),
        document_examples=("벤처기업인증서", "중소기업확인서", "사업자등록증", "법인등기부등본", "표준재무제표증명"),
        business_use_cases=("지원사업 증빙", "입찰 제출", "거래처 등록", "기업 자격 확인"),
        core_features=("certificate_score", "official_form_score", "business_registration_pattern", "centered_title_score", "seal_or_barcode_area_score"),
    ),
    DocumentPatternProfile(
        type="동의서",
        aliases=("개인정보 동의서", "Consent Form", "Agreement Form"),
        profile_text=(
            "동의서는 개인정보, 기업정보, 참여조건, 행사 참여 등에 대해 동의를 받기 위한 문서다. 개인정보 수집·이용 동의, 제3자 제공 동의, "
            "참여 동의서, 동의합니다, 수집·이용 목적, 수집 항목, 보유기간, 거부권, 서명 같은 표현이 자주 등장한다. "
            "항목별 안내문과 체크/동의 문구, 서명란이 포함되는 경우가 많다."
        ),
        tags=("동의", "개인정보", "서명"),
        semantic_signals=("동의서", "개인정보", "기업정보", "수집", "이용", "제3자 제공", "동의합니다", "보유기간", "거부권", "서명", "참여동의"),
        layout_signals=("안내문 + 항목 목록", "체크박스 가능", "서명란 가능", "긴 설명 문단"),
        structural_signals=("수집·이용 목적 → 수집 항목 → 보유기간 → 동의/서명", "개인정보/기업정보 관련 문구 반복"),
        ocr_signals=("동의", "개인정보", "수집", "이용", "제공", "서명"),
        numeric_patterns=("보유기간", "전화번호", "생년월일", "날짜"),
        document_examples=("개인정보 수집이용 동의서", "제3자 제공 동의서", "참여 동의서"),
        business_use_cases=("사업 참여 접수", "개인정보 처리", "행사/프로그램 참여"),
        core_features=("consent_form_score", "privacy_terms_count", "signature_area_score", "checkbox_pattern_score"),
    ),
)


def _all_document_pattern_profiles() -> tuple[DocumentPatternProfile, ...]:
    additional_profiles = tuple(profile for profile in ADDITIONAL_DOCUMENT_PATTERN_PROFILES if profile.type != "인증서")
    return _merge_pattern_profiles(
        DOCUMENT_PATTERN_PROFILES + additional_profiles + DATASET_DOCUMENT_PATTERN_PROFILES
    )


def _merge_pattern_profiles(profiles: tuple[DocumentPatternProfile, ...]) -> tuple[DocumentPatternProfile, ...]:
    merged: dict[str, DocumentPatternProfile] = {}
    order: list[str] = []
    for profile in profiles:
        if profile.type not in merged:
            merged[profile.type] = profile
            order.append(profile.type)
            continue
        previous = merged[profile.type]
        merged[profile.type] = DocumentPatternProfile(
            type=profile.type,
            aliases=_unique_tuple((*previous.aliases, *profile.aliases)),
            profile_text=profile.profile_text or previous.profile_text,
            tags=_unique_tuple((*previous.tags, *profile.tags)),
            semantic_signals=_unique_tuple((*previous.semantic_signals, *profile.semantic_signals)),
            layout_signals=_unique_tuple((*previous.layout_signals, *profile.layout_signals)),
            structural_signals=_unique_tuple((*previous.structural_signals, *profile.structural_signals)),
            ocr_signals=_unique_tuple((*previous.ocr_signals, *profile.ocr_signals)),
            numeric_patterns=_unique_tuple((*previous.numeric_patterns, *profile.numeric_patterns)),
            document_examples=_unique_tuple((*previous.document_examples, *profile.document_examples)),
            business_use_cases=_unique_tuple((*previous.business_use_cases, *profile.business_use_cases)),
            core_features=_unique_tuple((*previous.core_features, *profile.core_features)),
        )
    return tuple(merged[key] for key in order)


def _unique_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return tuple(result)


def get_default_document_patterns() -> list[dict[str, Any]]:
    return [
        {
            "type": profile.type,
            "profile_text": profile.profile_text,
            "tags": list(profile.tags),
            "profile_signals": profile.to_profile_signals(),
        }
        for profile in _all_document_pattern_profiles()
    ]


def get_pattern_for_type(document_type: str) -> dict[str, Any] | None:
    normalized = document_type.strip().lower()
    for profile in _all_document_pattern_profiles():
        labels = (profile.type, *profile.aliases)
        if any(normalized == label.strip().lower() for label in labels):
            return {
                "type": profile.type,
                "profile_text": profile.profile_text,
                "tags": list(profile.tags),
                "profile_signals": profile.to_profile_signals(),
            }
    return None


def build_evidence_groups(
    *,
    predicted_type: str,
    text: str,
    structural_features: dict[str, Any],
    layout_features: dict[str, Any],
    text_stats: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    profile = get_pattern_for_type(predicted_type)
    if profile is None:
        return {"semantic": [], "layout": [], "structure": [], "ocr": []}
    signals = profile["profile_signals"]
    lowered = text.lower()
    semantic = [
        {"signal": signal, "reason": "text_match"}
        for signal in signals.get("semantic_signals", [])
        if str(signal).lower() in lowered
    ][:8]
    ocr = [
        {"signal": signal, "reason": "ocr_text_match"}
        for signal in signals.get("ocr_signals", [])
        if str(signal).lower() in lowered
    ][:6]
    if float(text_stats.get("low_quality_scan_score", 0.0) or 0.0) >= 0.5:
        ocr.append({"signal": "low_quality_scan_score", "value": text_stats.get("low_quality_scan_score")})

    layout = _feature_evidence(
        layout_features,
        {
            "dense_text_score": 0.55,
            "receipt_pattern_score": 0.45,
            "slide_like_layout_score": 0.45,
            "two_column_score": 0.45,
            "signature_area_score": 0.35,
            "numeric_column_score": 0.35,
            "approval_block_score": 0.35,
            "chart_presence_score": 0.35,
            "large_header_score": 0.45,
        },
    )
    structure = _feature_evidence(
        structural_features,
        {
            "citation_count": 2.0,
            "contract_terms_count": 2.0,
            "receipt_terms_count": 2.0,
            "table_count": 1.0,
            "image_count": 1.0,
            "bullet_ratio": 0.2,
            "clause_pattern_score": 0.3,
            "legal_term_density": 0.2,
            "research_structure_score": 0.3,
            "heading_density": 0.2,
        },
    )
    return {"semantic": semantic, "layout": layout[:8], "structure": structure[:8], "ocr": ocr[:8]}


def _feature_evidence(features: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for key, threshold in thresholds.items():
        try:
            value = float(features.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value >= threshold:
            evidence.append({"feature": key, "value": round(value, 4), "threshold": threshold})
    return evidence
