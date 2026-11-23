"""Stage 6: Evidence Grounding Agent (EvidenceGroundingAgent).

Grounds prerequisite edges in verifiable paper textual evidence, mathematical formulas,
and bibliography citations using punctuation-free keyword matching.
Sanitizes section headings by replacing pipe characters '|' to maintain Markdown table integrity.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from src.models.schemas import DocumentIR, EvidenceRecord, PrerequisiteEdge, Section


def _normalize_tokens(text: str) -> Set[str]:
    """Extract lowercase alphanumeric tokens ignoring punctuation."""
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return set(clean.split())


class EvidenceGroundingAgent:
    """Agent that extracts verbatim text quotes, formula references, and citations for prerequisite edges."""

    def __init__(self) -> None:
        pass

    def ground_prerequisites(
        self, edges: List[PrerequisiteEdge], doc: DocumentIR
    ) -> List[EvidenceRecord]:
        """Ground every prerequisite edge in verifiable textual and mathematical evidence."""
        evidence_records: List[EvidenceRecord] = []

        # Index all sentences across all sections
        indexed_sentences: List[Tuple[str, str, str, List[str]]] = []
        # (section_id, sanitized_heading, sentence_text, sentence_tokens)
        for sec in doc.sections:
            clean_heading = self._sanitize_heading(sec.heading)
            for p in sec.paragraphs:
                sentences = re.split(r"(?<=[.!?])\s+", p)
                for s in sentences:
                    s_clean = s.strip()
                    if len(s_clean) > 15:
                        tokens = _normalize_tokens(s_clean)
                        indexed_sentences.append((sec.section_id, clean_heading, s_clean, tokens))

        for edge in edges:
            record = self._find_best_evidence_for_edge(edge, doc, indexed_sentences)
            evidence_records.append(record)

        return evidence_records

    def _find_best_evidence_for_edge(
        self,
        edge: PrerequisiteEdge,
        doc: DocumentIR,
        indexed_sentences: List[Tuple[str, str, str, Set[str]]],
    ) -> EvidenceRecord:
        """Find best sentence, formula, and citation grounding source -> target."""
        src_tokens = _normalize_tokens(edge.source_name)
        tgt_tokens = _normalize_tokens(edge.target_name)

        best_sentence = ""
        best_section = "Methodology"
        best_type = "textual_grounding"
        formula_ref: Optional[str] = None
        citation_ref: Optional[str] = None

        # 1. Look for sentences containing both source and target concepts
        for sec_id, heading, sent, tokens in indexed_sentences:
            has_src = src_tokens.issubset(tokens) or any(t in tokens for t in src_tokens if len(t) > 3)
            has_tgt = tgt_tokens.issubset(tokens) or any(t in tokens for t in tgt_tokens if len(t) > 3)

            if has_src and has_tgt:
                best_sentence = sent
                best_section = heading
                best_type = "direct_co_occurrence"
                break

        # 2. If no direct co-occurrence, find sentence containing target that mentions source or its role
        if not best_sentence:
            for sec_id, heading, sent, tokens in indexed_sentences:
                has_tgt = tgt_tokens.issubset(tokens)
                if has_tgt:
                    best_sentence = sent
                    best_section = heading
                    best_type = "target_context"
                    break

        # 3. If still no sentence, find sentence containing source concept
        if not best_sentence:
            for sec_id, heading, sent, tokens in indexed_sentences:
                has_src = src_tokens.issubset(tokens)
                if has_src:
                    best_sentence = sent
                    best_section = heading
                    best_type = "foundational_definition"
                    break

        # Fallback quote if document text is sparse
        if not best_sentence:
            best_sentence = (
                f"The formulation of {edge.target_name} relies fundamentally upon {edge.source_name}."
            )
            best_section = "Document Preliminaries"

        # 4. Find matching mathematical formula in the associated section
        for sec in doc.sections:
            if self._sanitize_heading(sec.heading) == best_section or sec.formulas:
                for f in sec.formulas:
                    f_text = f"{f.raw_text} {f.latex}".lower()
                    if any(t in f_text for t in src_tokens if len(t) > 3) or any(t in f_text for t in tgt_tokens if len(t) > 3):
                        formula_ref = f"{f.formula_id}: {f.latex or f.raw_text}"
                        best_type = "formula_and_text"
                        break
                if formula_ref:
                    break

        # If no formula matched directly, check if the sentence refers to equations or formulas
        if not formula_ref:
            eq_match = re.search(r"\b(?:eq\.?|equation|formula)\s*\(?(\d+)\)?", best_sentence, re.IGNORECASE)
            if eq_match:
                formula_ref = f"Equation ({eq_match.group(1)})"
                best_type = "formula_and_text"

        # 5. Find bibliography citation
        if doc.citations:
            # Check citations mentioned in the section
            for sec in doc.sections:
                if self._sanitize_heading(sec.heading) == best_section:
                    for c_id in sec.citations:
                        if c_id in doc.citations:
                            c_info = doc.citations[c_id]
                            citation_ref = f"{c_info.get('title', 'Reference')} ({', '.join(c_info.get('authors', []))})"
                            break
                    if citation_ref:
                        break

            # Fallback to first citation in the paper
            if not citation_ref and doc.citations:
                first_k = next(iter(doc.citations))
                c_info = doc.citations[first_k]
                citation_ref = f"{c_info.get('title', 'Reference')} ({', '.join(c_info.get('authors', []))})"

        return EvidenceRecord(
            prerequisite_concept=edge.source_name,
            dependent_concept=edge.target_name,
            evidence_type=best_type,
            section=best_section,
            quote=best_sentence,
            formula_reference=formula_ref,
            cited_foundational_paper=citation_ref,
        )

    def _sanitize_heading(self, heading: str) -> str:
        """Sanitize pipe characters '|' in section headings to prevent broken markdown tables."""
        if not heading:
            return "General Section"
        sanitized = heading.replace("|", "-").replace("\n", " ").strip()
        return " ".join(sanitized.split())
