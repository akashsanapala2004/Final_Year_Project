"""Stage 7: Pipeline Orchestrator (Orchestrator).

Coordinates Stages 1 through 6 of the Scientific Prerequisite System:
1. PaperParserAgent: Document & formula parsing
2. ConceptExtractorAgent: 4-way taxonomy concept extraction
3. PrerequisiteRelationAgent: Asymmetric RefD & iPRL relation learning
4. GraphReasoningAgent: DAG construction & FAS cycle removal
5. CurriculumPlannerAgent: Kahn's topological sort & 4-tier monotonic curriculum
6. EvidenceGroundingAgent: Grounded in-text quotes, formulas, and citations
Exports structured JSON, Markdown curriculum reports, and Mermaid diagrams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.agents.concept_extractor_agent import ConceptExtractorAgent
from src.agents.curriculum_planner_agent import CurriculumPlannerAgent
from src.agents.evidence_grounding_agent import EvidenceGroundingAgent
from src.agents.graph_reasoning_agent import GraphReasoningAgent
from src.agents.paper_parser_agent import PaperParserAgent
from src.agents.prerequisite_relation_agent import PrerequisiteRelationAgent
from src.models.schemas import (
    CurriculumPlan,
    DocumentIR,
    EvidenceRecord,
    PrerequisiteDAG,
)
from src.utils.graph_viz import MermaidGraphExporter


class PipelineOrchestrator:
    """Master Coordinator for end-to-end prerequisite graph extraction and curriculum planning."""

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.parser_agent = PaperParserAgent(data_dir=self.data_dir)
        self.concept_agent = ConceptExtractorAgent()
        self.relation_agent = PrerequisiteRelationAgent()
        self.graph_agent = GraphReasoningAgent()
        self.curriculum_agent = CurriculumPlannerAgent()
        self.evidence_agent = EvidenceGroundingAgent()
        self.viz_exporter = MermaidGraphExporter()

    def run_pipeline(
        self, paper_input: Union[str, Path], save_artifacts: bool = True
    ) -> Dict[str, Any]:
        """Execute full 6-stage pipeline on given paper ID or file path."""
        # Stage 1: Document & Formula Parsing
        doc: DocumentIR = self.parser_agent.parse(paper_input)

        # Stage 2: Concept Extraction
        concepts = self.concept_agent.extract_concepts(doc)

        # Stage 3: Asymmetric Prerequisite Learning
        edges = self.relation_agent.learn_prerequisites(concepts, doc)

        # Stage 4: Graph Reasoning, FAS Cycle Elimination & Transitive Reduction
        dag: PrerequisiteDAG = self.graph_agent.build_dag(doc.paper_id, concepts, edges)

        # Stage 5: Pedagogical Curriculum Planning with DAG Monotonicity
        curriculum: CurriculumPlan = self.curriculum_agent.plan_curriculum(dag, doc.title)

        # Stage 6: Grounded Evidence Generation
        evidence: List[EvidenceRecord] = self.evidence_agent.ground_prerequisites(dag.edges, doc)

        # Generate Mermaid diagram
        mermaid_code = self.viz_exporter.export(dag, curriculum)

        result = {
            "paper_id": doc.paper_id,
            "document": doc,
            "concepts": concepts,
            "edges": dag.edges,
            "dag": dag,
            "curriculum": curriculum,
            "evidence": evidence,
            "mermaid": mermaid_code,
            "paths": {},
        }

        if save_artifacts:
            saved_paths = self._save_outputs(doc, dag, curriculum, evidence, mermaid_code)
            result["paths"] = saved_paths

        return result

    def _save_outputs(
        self,
        doc: DocumentIR,
        dag: PrerequisiteDAG,
        curriculum: CurriculumPlan,
        evidence: List[EvidenceRecord],
        mermaid_code: str,
    ) -> Dict[str, Path]:
        """Save JSON, Markdown report, and Mermaid files to output directory."""
        paper_id = doc.paper_id

        # 1. Save JSON
        json_path = self.output_dir / f"{paper_id}_prerequisite_curriculum.json"
        combined_payload = {
            "paper_id": paper_id,
            "title": doc.title,
            "abstract": doc.abstract,
            "metrics": dag.metrics.to_dict(),
            "nodes": [n.to_dict() for n in dag.nodes],
            "edges": [e.to_dict() for e in dag.edges],
            "curriculum": curriculum.to_dict(),
            "evidence": [ev.to_dict() for ev in evidence],
        }
        json_path.write_text(json.dumps(combined_payload, indent=2), encoding="utf-8")

        # 2. Save Mermaid Diagram
        mmd_path = self.output_dir / f"{paper_id}_graph.mmd"
        mmd_path.write_text(mermaid_code, encoding="utf-8")

        # 3. Save Markdown Report
        md_path = self.output_dir / f"{paper_id}_curriculum_report.md"
        report_md = self._generate_markdown_report(doc, dag, curriculum, evidence, mermaid_code)
        md_path.write_text(report_md, encoding="utf-8")

        return {
            "json": json_path,
            "mermaid": mmd_path,
            "markdown": md_path,
        }

    def _generate_markdown_report(
        self,
        doc: DocumentIR,
        dag: PrerequisiteDAG,
        curriculum: CurriculumPlan,
        evidence: List[EvidenceRecord],
        mermaid_code: str,
    ) -> str:
        """Format a comprehensive, publication-grade Markdown curriculum report."""
        lines: List[str] = []
        lines.append(f"# Prerequisite Curriculum Report: {doc.title}")
        lines.append(f"**Paper ID:** `{doc.paper_id}`  ")
        lines.append(f"**Total Foundational Prerequisites:** `{curriculum.total_prerequisites}`  ")
        lines.append(f"**Graph Acyclicity (DAG):** `{'Verified Strict DAG' if dag.metrics.is_dag else 'Non-DAG'}`  ")
        lines.append(f"**Cycles Resolved (Greedy FAS):** `{dag.metrics.cycles_detected_and_resolved}`  ")
        lines.append(f"**Graph Density:** `{dag.metrics.density}`\n")

        lines.append("## Abstract")
        lines.append(f"> {doc.abstract or 'No abstract provided in source document.'}\n")

        lines.append("## Prerequisite Knowledge Graph (DAG)")
        lines.append("```mermaid")
        lines.append(mermaid_code.strip())
        lines.append("```\n")

        lines.append("## Pedagogical Learning Sequence (4 Tiers)")
        for tier in curriculum.curriculum:
            lines.append(f"### {tier.tier_name}")
            if not tier.concepts:
                lines.append("*No concepts assigned to this tier.*\n")
                continue

            for idx, c in enumerate(tier.concepts, 1):
                role = "**[Target Contribution]**" if c.is_target_paper_contribution else f"({c.category})"
                lines.append(f"{idx}. **{c.name}** {role}")
                lines.append(f"   - **Why Needed:** {c.why_needed}")
                if c.prerequisites:
                    lines.append(f"   - **Direct Prerequisites:** {', '.join(c.prerequisites)}")
                else:
                    lines.append("   - **Direct Prerequisites:** None (Root Foundation)")
            lines.append("")

        lines.append("## Verifiable Evidence Grounding")
        lines.append("| Prerequisite Concept | Dependent Concept | Section | Evidence Type | Verbatim Quote / Formula |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")

        for ev in evidence:
            clean_quote = ev.quote.replace("|", "-").replace("\n", " ").strip()
            if len(clean_quote) > 120:
                clean_quote = clean_quote[:117] + "..."
            formula_note = f" (Formula: {ev.formula_reference})" if ev.formula_reference else ""
            lines.append(
                f"| {ev.prerequisite_concept} | {ev.dependent_concept} | {ev.section} | {ev.evidence_type} | \"{clean_quote}\"{formula_note} |"
            )

        lines.append("")
        if doc.citations:
            lines.append("## Foundational References Cited")
            for c_id, c_data in doc.citations.items():
                authors = ", ".join(c_data.get("authors", [])) or "Unknown Authors"
                year = f" ({c_data['date']})" if c_data.get("date") else ""
                lines.append(f"- **[{c_id}]** {authors}{year}. *{c_data.get('title', 'Paper Reference')}*. {c_data.get('venue', '')}")
            lines.append("")

        return "\n".join(lines) + "\n"
