#!/usr/bin/env python3
"""Search Catalog Utility (query.py).

Provides fast search across the paper library:
    python3 query.py --category Mathematics
    python3 query.py --concept "GLRT" --details
Outputs matching papers, concept roles, exact file paths (PDF, XML, JSON, MD), and grounded in-text quotes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_corpus_data(data_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """Load all processed paper JSONs from output directory."""
    records: List[Dict[str, Any]] = []
    if not output_dir.is_dir():
        return records

    for jf in sorted(output_dir.glob("*_prerequisite_curriculum.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            pid = data.get("paper_id", jf.stem.replace("_prerequisite_curriculum", ""))

            # Resolve associated file paths
            pdf_path = data_dir / "raw_pdfs" / f"{pid}.pdf"
            tei_path1 = data_dir / "grobid_output" / f"{pid}.tei.xml"
            tei_path2 = data_dir / "grobid_output" / f"{pid}.xml"
            tei_path = tei_path1 if tei_path1.is_file() else (tei_path2 if tei_path2.is_file() else None)
            md_path = output_dir / f"{pid}_curriculum_report.md"

            data["paths"] = {
                "json": jf.resolve(),
                "markdown": md_path.resolve() if md_path.is_file() else None,
                "xml": tei_path.resolve() if tei_path else None,
                "pdf": pdf_path.resolve() if pdf_path.is_file() else None,
            }
            records.append(data)
        except Exception:
            continue

    return records


def search_by_category(records: List[Dict[str, Any]], category: str) -> None:
    """Search and display all concepts in a given taxonomic category."""
    cat_clean = category.strip().lower()
    print("=" * 80)
    print(f"SEARCH RESULTS FOR CATEGORY: '{category.capitalize()}'")
    print("=" * 80)

    found_count = 0
    for rec in records:
        pid = rec.get("paper_id")
        title = rec.get("title", pid)
        paths = rec.get("paths", {})

        matching_concepts = []
        curriculum = rec.get("curriculum", {}).get("curriculum", [])
        for tier in curriculum:
            for c in tier.get("concepts", []):
                if c.get("category", "").lower() == cat_clean:
                    matching_concepts.append((tier.get("tier"), tier.get("tier_name"), c))

        if matching_concepts:
            found_count += len(matching_concepts)
            print(f"\n[Paper: {title}] (ID: {pid})")
            print(f"  • JSON: {paths.get('json')}")
            if paths.get("markdown"):
                print(f"  • MD:   {paths.get('markdown')}")
            if paths.get("xml"):
                print(f"  • XML:  {paths.get('xml')}")
            if paths.get("pdf"):
                print(f"  • PDF:  {paths.get('pdf')}")

            for t_num, t_name, c in matching_concepts:
                role = "Target Paper Contribution" if c.get("is_target_paper_contribution") else "Foundational Prerequisite"
                print(f"    - {c.get('name')} [Tier {t_num} | Role: {role}]")
                print(f"      Why Needed: {c.get('why_needed')}")

    if found_count == 0:
        print(f"No concepts found matching category '{category}'.")
        print("Valid categories: Mathematics, Algorithms, Theories, Methods.")
        print("Note: Ensure papers have been processed first via 'python3 run.py --all'.")
    else:
        print("\n" + "=" * 80)
        print(f"Found {found_count} concept occurrences across {len(records)} papers.")
        print("=" * 80)


def search_by_concept(records: List[Dict[str, Any]], concept_query: str, show_details: bool) -> None:
    """Search and display concept mentions, roles, files, and grounded evidence quotes."""
    q = concept_query.strip().lower()
    print("=" * 80)
    print(f"SEARCH RESULTS FOR CONCEPT: '{concept_query}' (Details: {'Enabled' if show_details else 'Disabled'})")
    print("=" * 80)

    found_count = 0
    for rec in records:
        pid = rec.get("paper_id")
        title = rec.get("title", pid)
        paths = rec.get("paths", {})
        curriculum = rec.get("curriculum", {}).get("curriculum", [])
        evidence_list = rec.get("evidence", [])

        # Find matching concepts in nodes or curriculum
        matched_in_paper = []
        for tier in curriculum:
            for c in tier.get("concepts", []):
                name = c.get("name", "")
                if q in name.lower() or name.lower() in q:
                    matched_in_paper.append((tier.get("tier"), tier.get("tier_name"), c))

        if matched_in_paper:
            found_count += 1
            print(f"\n[+] Paper: {title}")
            print(f"    Paper ID: {pid}")
            print(f"    Exact File Paths:")
            print(f"      • JSON Report: {paths.get('json')}")
            if paths.get("markdown"):
                print(f"      • Markdown:    {paths.get('markdown')}")
            if paths.get("xml"):
                print(f"      • TEI-XML:     {paths.get('xml')}")
            if paths.get("pdf"):
                print(f"      • PDF Source:  {paths.get('pdf')}")

            for t_num, t_name, c in matched_in_paper:
                role = "Target Paper Contribution" if c.get("is_target_paper_contribution") else "Foundational Prerequisite"
                print(f"\n    Concept: {c.get('name')} ({c.get('category')})")
                print(f"    Tier:    {t_name}")
                print(f"    Role:    {role}")
                print(f"    Why:     {c.get('why_needed')}")
                if c.get("cso_uri"):
                    print(f"    CSO URI: {c.get('cso_uri')} (Topic ID: {c.get('cso_topic_id')})")
                prereqs = c.get("prerequisites", [])
                if prereqs:
                    print(f"    Prereqs: {', '.join(prereqs)}")

            if show_details:
                print(f"\n    Grounded Textual Evidence & Formulas:")
                matched_evidence = [
                    ev for ev in evidence_list
                    if q in ev.get("prerequisite_concept", "").lower() or q in ev.get("dependent_concept", "").lower()
                ]
                if matched_evidence:
                    for ev in matched_evidence:
                        print(f"      • Relation: {ev.get('prerequisite_concept')} -> {ev.get('dependent_concept')}")
                        print(f"        Section:  {ev.get('section')}")
                        print(f"        Type:     {ev.get('evidence_type')}")
                        print(f"        Quote:    \"{ev.get('quote')}\"")
                        if ev.get("formula_reference"):
                            print(f"        Formula:  {ev.get('formula_reference')}")
                        if ev.get("cited_foundational_paper"):
                            print(f"        Citation: {ev.get('cited_foundational_paper')}")
                else:
                    print("      (No specific pairwise evidence record matched query)")

    if found_count == 0:
        print(f"No papers found containing concept '{concept_query}'.")
        print("Note: Run 'python3 run.py --all' first to extract concepts from all papers.")
    else:
        print("\n" + "=" * 80)
        print(f"Concept '{concept_query}' found in {found_count} papers.")
        print("=" * 80)


def search_cso_ontology(query: str, records: List[Dict[str, Any]]) -> None:
    """Classify and identify concepts directly using CSO.3.3.owl ontology."""
    from src.utils.cso_classifier import get_cso_ontology

    cso = get_cso_ontology()
    topic = cso.identify(query)

    print("=" * 80)
    print(f"CSO ONTOLOGY IDENTIFICATION & CLASSIFICATION: '{query}'")
    print("=" * 80)

    if not topic:
        # Fallback to direct classification rule
        cat = cso.classify(query, query)
        print(f"Term:           {query}")
        print(f"Ontology Match: No exact topic found in CSO.3.3.owl (14,416 topics)")
        print(f"Classification: {cat.value}")
    else:
        print(f"Canonical Label:  {topic.label}")
        print(f"Topic ID:         {topic.topic_id}")
        print(f"Ontology URI:     {topic.uri}")
        print(f"Classification:   {topic.category.value}")
        if topic.super_topics:
            print(f"Super-Topics:     {', '.join(topic.super_topics[:5])}")
        if topic.sub_topics:
            print(f"Sub-Topics:       {', '.join(topic.sub_topics[:8])}")
        if topic.equivalents:
            print(f"Equivalents:      {', '.join(topic.equivalents[:5])}")

    # Check occurrence in analyzed paper corpus
    print("\n--- Occurrences Across Analyzed Papers ---")
    matched_papers = []
    q_low = query.strip().lower()
    for rec in records:
        curriculum = rec.get("curriculum", {}).get("curriculum", [])
        for tier in curriculum:
            for c in tier.get("concepts", []):
                c_name = c.get("name", "").lower()
                cso_tid = (c.get("cso_topic_id") or "").lower()
                if q_low in c_name or (topic and topic.topic_id.lower() == cso_tid):
                    matched_papers.append((rec.get("title"), c.get("name"), tier.get("tier_name")))

    if matched_papers:
        for p_title, c_name, t_name in matched_papers:
            print(f"  • In '{p_title}': as '{c_name}' [{t_name}]")
    else:
        print("  (Not currently extracted in loaded paper library)")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Academic Concept Search and Catalog Utility")
    parser.add_argument("--category", "-c", type=str, help="Search by category (Mathematics, Algorithms, Theories, Methods)")
    parser.add_argument("--concept", "-k", type=str, help="Search by concept name or acronym (e.g. GLRT, Oblique Projection)")
    parser.add_argument("--cso", type=str, help="Identify and classify concept directly via CSO.3.3.owl ontology")
    parser.add_argument("--details", "-d", action="store_true", help="Show deep grounded quotes, formulas, and citations")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing data")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory containing output JSONs")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    records = load_corpus_data(data_dir, output_dir)

    if args.cso:
        search_cso_ontology(args.cso, records)
        return 0

    if args.category:
        search_by_category(records, args.category)
        return 0

    if args.concept:
        search_by_concept(records, args.concept, args.details)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
