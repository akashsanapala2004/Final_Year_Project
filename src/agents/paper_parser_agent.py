"""Stage 1: Document & Formula Parser Agent (PaperParserAgent).

Parses academic research papers from GROBID TEI-XML or raw document inputs.
Rejects copyright boilerplates in title, parses formulas, sections, and citations into DocumentIR.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.models.schemas import DocumentIR, Formula, Section

COPYRIGHT_PATTERNS = [
    re.compile(r"permission\s+to\s+make\s+digital", re.IGNORECASE),
    re.compile(r"is\s+granted\s+without\s+fee", re.IGNORECASE),
    re.compile(r"copyright\s+(\(c\)|©|\d{4})", re.IGNORECASE),
    re.compile(r"all\s+rights\s+reserved", re.IGNORECASE),
    re.compile(r"acm\s+reference\s+format", re.IGNORECASE),
    re.compile(r"proceedings\s+of\s+the", re.IGNORECASE),
    re.compile(r"this\s+paper\s+is\s+published\s+under", re.IGNORECASE),
    re.compile(r"creative\s+commons\s+attribution", re.IGNORECASE),
    re.compile(r"ieee\s+transactions\s+on", re.IGNORECASE),
]

PRELIMINARY_HEADING_KEYWORDS = [
    "preliminar",
    "background",
    "foundation",
    "system model",
    "problem formulation",
    "signal model",
    "mathematical model",
    "theoretical framework",
    "notation",
]


def _clean_tag(tag: str) -> str:
    """Strip XML namespace prefix if present."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _extract_all_text(elem: Optional[ET.Element]) -> str:
    """Recursively extract and clean text from an XML element."""
    if elem is None:
        return ""
    text_parts = []
    for text in elem.itertext():
        if text:
            text_parts.append(text)
    joined = " ".join("".join(text_parts).split())
    return joined.strip()


def is_copyright_boilerplate(text: str) -> bool:
    """Check if the given text matches copyright disclaimer patterns."""
    if not text or len(text.strip()) < 5:
        return False
    clean = text.strip()
    for pattern in COPYRIGHT_PATTERNS:
        if pattern.search(clean):
            return True
    return False


class PaperParserAgent:
    """Agent responsible for parsing academic papers into DocumentIR."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path("data")
        self.grobid_dir = self.data_dir / "grobid_output"

    def resolve_path(self, paper_input: Union[str, Path]) -> Path:
        """Resolve paper input to an existing XML file, converting from PDF if needed."""
        p = Path(paper_input)
        if p.is_file():
            if p.suffix.lower() == ".pdf":
                from src.utils.pdf_grobid_converter import PDFGrobidConverter
                converter = PDFGrobidConverter(output_dir=self.grobid_dir)
                return converter.convert_pdf_to_tei(p)
            return p

        # Check in grobid_dir
        candidates = [
            self.grobid_dir / str(paper_input),
            self.grobid_dir / f"{paper_input}.tei.xml",
            self.grobid_dir / f"{paper_input}.xml",
            self.grobid_dir / f"{p.stem}.tei.xml",
            self.grobid_dir / f"{p.stem}.xml",
            Path(f"data/grobid_output/{paper_input}.tei.xml"),
            Path(f"data/grobid_output/{paper_input}.xml"),
        ]
        for c in candidates:
            if c.is_file():
                try:
                    ET.parse(c)
                    return c
                except Exception:
                    pass  # If file is invalid XML, proceed to check PDF sources

        # Check if corresponding PDF exists in raw_pdfs
        pdf_candidates = [
            self.data_dir / "raw_pdfs" / str(paper_input),
            self.data_dir / "raw_pdfs" / f"{paper_input}.pdf",
            self.data_dir / "raw_pdfs" / f"{p.stem}.pdf",
            self.data_dir / "raw_pdfs" / p.name,
            Path(f"data/raw_pdfs/{paper_input}.pdf"),
            Path(f"data/raw_pdfs/{p.stem}.pdf"),
            Path(f"data/raw_pdfs/{p.name}"),
        ]
        for pdf_c in pdf_candidates:
            if pdf_c.is_file():
                from src.utils.pdf_grobid_converter import PDFGrobidConverter
                converter = PDFGrobidConverter(output_dir=self.grobid_dir)
                return converter.convert_pdf_to_tei(pdf_c, force_recompute=True)

        raise FileNotFoundError(f"Could not resolve paper XML or PDF for input: {paper_input}")

    def parse(self, paper_input: Union[str, Path]) -> DocumentIR:
        """Parse GROBID TEI-XML or raw XML text into DocumentIR."""
        if isinstance(paper_input, str) and paper_input.strip().startswith("<"):
            root = ET.fromstring(paper_input)
            paper_id = "inline_xml"
        else:
            file_path = self.resolve_path(paper_input)
            tree = ET.parse(file_path)
            root = tree.getroot()
            paper_id = file_path.stem.replace(".tei", "")

        return self._parse_tei_tree(root, paper_id)

    def _parse_tei_tree(self, root: ET.Element, paper_id: str) -> DocumentIR:
        """Extract title, abstract, sections, formulas, and citations from TEI-XML root."""
        # 1. Parse Title with copyright rejection
        title = self._extract_clean_title(root, paper_id)

        # 2. Parse Abstract
        abstract = self._extract_abstract(root)

        # 3. Parse Citations bibliography
        citations = self._extract_citations(root)

        # 4. Parse Sections & Formulas
        sections, preliminaries = self._extract_sections_and_formulas(root)

        return DocumentIR(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            sections=sections,
            preliminaries=preliminaries,
            citations=citations,
        )

    def _extract_clean_title(self, root: ET.Element, fallback_id: str) -> str:
        """Extract title rejecting copyright disclaimers (Anti-Pattern #4)."""
        candidate_titles: List[str] = []

        # Look in teiHeader -> titleStmt -> title
        for elem in root.iter():
            tag = _clean_tag(elem.tag)
            if tag == "title":
                t = _extract_all_text(elem)
                if t and not is_copyright_boilerplate(t):
                    candidate_titles.append(t)

        for candidate in candidate_titles:
            # Check length and quality
            if len(candidate) > 5 and not is_copyright_boilerplate(candidate):
                return candidate

        # Fallback to first non-boilerplate section header or heading
        for elem in root.iter():
            tag = _clean_tag(elem.tag)
            if tag == "head":
                h = _extract_all_text(elem)
                if h and not is_copyright_boilerplate(h) and len(h) > 5:
                    return h

        # Fallback to first paragraph text snippet if not boilerplate
        for elem in root.iter():
            tag = _clean_tag(elem.tag)
            if tag == "p":
                p = _extract_all_text(elem)
                if p and not is_copyright_boilerplate(p):
                    # Truncate first sentence
                    first_sent = p.split(".")[0]
                    if len(first_sent) > 5:
                        return first_sent

        return fallback_id.replace("_", " ").title()

    def _extract_abstract(self, root: ET.Element) -> str:
        """Extract abstract text."""
        abstract_paragraphs = []
        for elem in root.iter():
            tag = _clean_tag(elem.tag)
            if tag == "abstract":
                for p in elem.iter():
                    if _clean_tag(p.tag) == "p":
                        t = _extract_all_text(p)
                        if t and not is_copyright_boilerplate(t):
                            abstract_paragraphs.append(t)
        return " ".join(abstract_paragraphs).strip()

    def _extract_citations(self, root: ET.Element) -> Dict[str, Any]:
        """Extract bibliography listBibl metadata."""
        citations: Dict[str, Any] = {}
        for elem in root.iter():
            tag = _clean_tag(elem.tag)
            if tag == "biblStruct":
                # Get xml:id or target id
                bib_id = ""
                for k, v in elem.attrib.items():
                    if k.endswith("id") or k == "id":
                        bib_id = v
                        break
                if not bib_id:
                    bib_id = f"bibr_{len(citations)}"

                title = ""
                authors = []
                date = ""
                venue = ""

                for sub in elem.iter():
                    sub_tag = _clean_tag(sub.tag)
                    if sub_tag == "title" and not title:
                        title = _extract_all_text(sub)
                    elif sub_tag in ("author", "persName"):
                        a_name = _extract_all_text(sub)
                        if a_name and a_name not in authors:
                            authors.append(a_name)
                    elif sub_tag == "date" and not date:
                        date = sub.attrib.get("when", _extract_all_text(sub))
                    elif sub_tag == "journal" or (sub_tag == "title" and sub.attrib.get("level") == "j"):
                        venue = _extract_all_text(sub)

                citations[bib_id] = {
                    "citation_id": bib_id,
                    "title": title or "Foundational Reference",
                    "authors": authors,
                    "date": date,
                    "venue": venue,
                }
        return citations

    def _extract_sections_and_formulas(
        self, root: ET.Element
    ) -> Tuple[List[Section], List[str]]:
        """Extract sections, headings, paragraphs, and formulas preserving formula IDs."""
        sections: List[Section] = []
        preliminaries: List[str] = []
        sec_counter = 0

        # Traverse body divs
        body_elem = None
        for elem in root.iter():
            if _clean_tag(elem.tag) == "body":
                body_elem = elem
                break

        div_elements = []
        if body_elem is not None:
            for child in body_elem:
                if _clean_tag(child.tag) == "div":
                    div_elements.append(child)

        if not div_elements:
            # Fallback to any div in the document
            for elem in root.iter():
                if _clean_tag(elem.tag) == "div":
                    div_elements.append(elem)

        formula_global_idx = 1
        for div in div_elements:
            sec_counter += 1
            sec_id = f"sec_{sec_counter}"
            for k, v in div.attrib.items():
                if k.endswith("id") or k == "id" or k == "n":
                    sec_id = f"sec_{v}"
                    break

            heading = ""
            paragraphs: List[str] = []
            formulas: List[Formula] = []
            sec_citations: List[str] = []

            for child in div:
                child_tag = _clean_tag(child.tag)
                if child_tag == "head":
                    h = _extract_all_text(child)
                    if not is_copyright_boilerplate(h):
                        heading = h
                elif child_tag == "p":
                    p_text = _extract_all_text(child)
                    if p_text and not is_copyright_boilerplate(p_text):
                        paragraphs.append(p_text)
                    # Extract citation refs inside this paragraph
                    for ref in child.iter():
                        if _clean_tag(ref.tag) == "ref":
                            target = ref.attrib.get("target", "").lstrip("#")
                            if target and target not in sec_citations:
                                sec_citations.append(target)
                elif child_tag == "formula":
                    f_id = ""
                    for k, v in child.attrib.items():
                        if k.endswith("id") or k == "id":
                            f_id = v
                            break
                    if not f_id:
                        f_id = f"formula_{formula_global_idx}"
                    formula_global_idx += 1
                    raw_f = _extract_all_text(child)
                    latex_f = child.attrib.get("latex", raw_f)
                    formulas.append(
                        Formula(
                            formula_id=f_id,
                            raw_text=raw_f,
                            latex=latex_f,
                            section_id=sec_id,
                        )
                    )

            # Check if any formulas are nested inside paragraphs
            for p_elem in div.iter():
                if _clean_tag(p_elem.tag) == "formula":
                    raw_f = _extract_all_text(p_elem)
                    if not any(f.raw_text == raw_f for f in formulas):
                        f_id = f"formula_{formula_global_idx}"
                        formula_global_idx += 1
                        formulas.append(
                            Formula(
                                formula_id=f_id,
                                raw_text=raw_f,
                                latex=p_elem.attrib.get("latex", raw_f),
                                section_id=sec_id,
                            )
                        )

            # Determine section category
            sec_cat = self._categorize_heading(heading)

            section_obj = Section(
                section_id=sec_id,
                heading=heading or f"Section {sec_counter}",
                category=sec_cat,
                paragraphs=paragraphs,
                formulas=formulas,
                citations=sec_citations,
            )
            sections.append(section_obj)

            # Check if preliminary
            if any(k in (heading or "").lower() for k in PRELIMINARY_HEADING_KEYWORDS):
                preliminaries.extend(paragraphs)

        return sections, preliminaries

    def _categorize_heading(self, heading: str) -> str:
        """Classify section heading into standardized scientific category."""
        h = heading.lower()
        if any(k in h for k in ["intro", "overview"]):
            return "Introduction"
        if any(k in h for k in PRELIMINARY_HEADING_KEYWORDS):
            return "Preliminaries"
        if any(k in h for k in ["method", "approach", "algorithm", "scheme", "framework", "proposed"]):
            return "Methodology"
        if any(k in h for k in ["experiment", "evaluation", "result", "simulation", "performance"]):
            return "Evaluation"
        if any(k in h for k in ["conclu", "summary", "future work"]):
            return "Conclusion"
        if any(k in h for k in ["related work", "literature", "prior work"]):
            return "Related Work"
        return "Body"
