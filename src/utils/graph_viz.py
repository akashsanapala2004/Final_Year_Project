"""Mermaid Flowchart Generator and Visualizer for Prerequisite DAGs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.models.schemas import CurriculumPlan, PrerequisiteDAG


def _clean_mermaid_id(name: str) -> str:
    """Convert concept name or ID to valid Mermaid node ID."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")
    return f"node_{clean}"


def _clean_mermaid_label(label: str) -> str:
    """Sanitize label string for safe rendering inside Mermaid quotes."""
    return label.replace('"', '\\"').replace("\n", " ").strip()


class MermaidGraphExporter:
    """Exports PrerequisiteDAG and CurriculumPlan to rich GitHub Flavored Mermaid Flowcharts."""

    def __init__(self) -> None:
        pass

    def export(self, dag: PrerequisiteDAG, plan: Optional[CurriculumPlan] = None) -> str:
        """Generate Mermaid flowchart string with tiered subgraphs and color styling."""
        lines: List[str] = ["flowchart TD"]

        # Tier node grouping if plan is provided
        tier_concept_map: Dict[str, int] = {}
        if plan:
            for tier in plan.curriculum:
                for c in tier.concepts:
                    tier_concept_map[c.concept_id] = tier.tier

        # Define styles
        lines.append("    %% Styling Definitions")
        lines.append("    classDef tier1 fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1;")
        lines.append("    classDef tier2 fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20;")
        lines.append("    classDef tier3 fill:#FFF3E0,stroke:#EF6C00,stroke-width:2px,color:#E65100;")
        lines.append("    classDef tier4 fill:#FCE4EC,stroke:#C2185B,stroke-width:3px,color:#880E4F;")

        # Nodes grouped into subgraphs if plan is available
        if plan:
            for tier in plan.curriculum:
                if not tier.concepts:
                    continue
                safe_title = _clean_mermaid_label(tier.tier_name)
                lines.append(f'    subgraph Tier{tier.tier} ["{safe_title}"]')
                for c in tier.concepts:
                    node_id = _clean_mermaid_id(c.concept_id)
                    label = _clean_mermaid_label(c.name)
                    lines.append(f'        {node_id}["{label}"]:::tier{tier.tier}')
                lines.append("    end")
        else:
            for node in dag.nodes:
                node_id = _clean_mermaid_id(node.concept_id)
                label = _clean_mermaid_label(node.name)
                tier_idx = 4 if node.is_target_paper_contribution else 1
                lines.append(f'    {node_id}["{label}"]:::tier{tier_idx}')

        lines.append("")
        lines.append("    %% Prerequisite Dependency Edges")

        # Edges
        for edge in dag.edges:
            src_id = _clean_mermaid_id(edge.source_id)
            tgt_id = _clean_mermaid_id(edge.target_id)
            if edge.asymmetry_score > 0:
                score_str = f"asym: {edge.asymmetry_score:.2f}"
                lines.append(f'    {src_id} -->|"{score_str}"| {tgt_id}')
            else:
                lines.append(f"    {src_id} --> {tgt_id}")

        return "\n".join(lines) + "\n"

    def save(
        self,
        dag: PrerequisiteDAG,
        plan: Optional[CurriculumPlan],
        filepath: Union[str, Path],
    ) -> Path:
        """Write Mermaid flowchart string to output file."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.export(dag, plan)
        p.write_text(content, encoding="utf-8")
        return p
