"""Stage 3: Prerequisite Relation Agent (PrerequisiteRelationAgent).

Implements Asymmetric Prerequisite Learning operationalizing:
1. Manrique et al. (2019): Reflected Reference Distance (RefD) for directional asymmetry.
2. Pan et al. (AAAI 2019 / iPRL framework): Taxonomic delta, sequential offset, lexical subsumption,
   and epistemic rationale generation.
Strict rule: A target paper contribution can NEVER precede foundational external prerequisites.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from src.models.schemas import (
    ConceptCategory,
    ConceptEntity,
    DocumentIR,
    PrerequisiteEdge,
)

# Taxonomic Hierarchy Levels: Math (1) < Theories (2) < Algorithms (3) < Methods (4)
TAXONOMIC_TIERS: Dict[ConceptCategory, float] = {
    ConceptCategory.MATHEMATICS: 1.0,
    ConceptCategory.THEORIES: 2.0,
    ConceptCategory.ALGORITHMS: 3.0,
    ConceptCategory.METHODS: 4.0,
}

# Domain foundational dependencies (Ground-truth epistemic relationships)
DOMAIN_DEPENDENCIES: List[Tuple[str, str, str]] = [
    ("Linear Algebra", "Subspaces", "Subspaces are vector subspaces defined within linear algebra vector spaces."),
    ("Linear Algebra", "Covariance Matrices", "Covariance matrices are positive semi-definite linear operators on multivariate data."),
    ("Linear Algebra", "Eigen-Decomposition", "Eigen-decomposition factorizes square matrices using linear algebraic basis transformation."),
    ("Linear Algebra", "Singular Value Decomposition", "SVD generalizes eigen-decomposition for arbitrary rectangular matrices."),
    ("Linear Algebra", "Pseudo-Inverse", "Moore-Penrose pseudo-inverse resolves ill-posed linear equations via linear algebra."),
    ("Linear Algebra", "Kronecker Products", "Kronecker products form tensor matrix products in multilinear algebra."),
    ("Subspaces", "Oblique Projection", "Oblique projection operators project vectors onto a target subspace along a reference subspace."),
    ("Subspaces", "Orthogonal Projection", "Orthogonal projection requires orthogonal complement subspaces."),
    ("Subspaces", "GLRT", "GLRT projects observed signal vectors into signal and noise subspaces to formulate decision statistics."),
    ("Oblique Projection", "Jamming Cancellation", "Oblique projection decomposes received signals into interference-free subspaces for jamming cancellation."),
    ("Covariance Matrices", "GLRT", "GLRT evaluates likelihood ratios parameterized by sample covariance matrices."),
    ("Hypothesis Testing", "GLRT", "GLRT is an algorithmic statistical decision rule derived from binary hypothesis testing."),
    ("Fourier Transform", "FFT", "FFT is an efficient algorithmic implementation of the discrete Fourier transform."),
    ("Fourier Transform", "IFFT", "IFFT algorithmically evaluates the inverse discrete Fourier transform."),
    ("IFFT", "OFDM", "OFDM multicarrier modulation uses IFFT to synthesize orthogonal subcarriers in the time domain."),
    ("FFT", "OFDM", "OFDM demodulation applies FFT to transform received time-domain samples back to frequency subcarriers."),
    ("Cognitive Radio", "Spectrum Sensing", "Spectrum sensing is the primary theoretical capability enabling opportunistic spectrum access in cognitive radio."),
    ("Spectrum Sensing", "GLRT", "GLRT provides the optimal algorithmic detector for binary spectrum sensing hypotheses."),
    ("Graph Theory", "Hypertree Decomposition", "Hypertree decomposition is a structural graph decomposition method for hypergraphs."),
    ("Graph Theory", "Graph Traversal", "Graph traversal (DFS/BFS) algorithmically explores vertices and edges defined in graph theory."),
    ("Hypertree Decomposition", "Hypergraph Reduction", "Hypertree decomposition provides the structural tree backbone for hypergraph reduction."),
    ("Case-Based Reasoning", "Hypergraph Reduction", "CBR retrieves past hypergraph reductions to guide polynomial-time subproblem solutions."),
    ("Computational Complexity", "Hypertree Decomposition", "Hypertree decomposition resolves NP-hard constraint satisfaction problems into polynomial-time tractable subproblems."),
    ("Convex Optimization", "Gradient Descent", "Gradient descent provides first-order iterative steps to reach global minima in convex optimization."),
    ("Boolean Logic", "Model Checking and Formal Verification", "Formal verification uses Boolean logic and temporal logic to model state safety properties."),
    ("Physical Layer Security", "Jamming Cancellation", "Jamming cancellation prevents adversarial jamming to maintain physical layer secrecy capacity."),
    ("Physical Layer Security", "OFDM", "OFDM channels provide frequency diversity utilized for physical layer secrecy."),
]


class PrerequisiteRelationAgent:
    """Agent that learns directional prerequisite relations between extracted concepts."""

    def __init__(self, confidence_threshold: float = 0.50) -> None:
        self.confidence_threshold = confidence_threshold
        self.taxonomic_tiers = TAXONOMIC_TIERS

    def learn_prerequisites(
        self, concepts: List[ConceptEntity], doc: DocumentIR
    ) -> List[PrerequisiteEdge]:
        """Learn asymmetric prerequisite edges A -> B across all concepts."""
        edges: List[PrerequisiteEdge] = []
        if len(concepts) < 2:
            return edges

        concept_by_name = {c.name.lower(): c for c in concepts}
        section_indices = {s.section_id: idx for idx, s in enumerate(doc.sections)}

        # Build pair candidates
        for i, source in enumerate(concepts):
            for j, target in enumerate(concepts):
                if i == j:
                    continue

                # STRICT RULE: A target paper contribution can NEVER precede foundational external prerequisites!
                if source.is_target_paper_contribution and not target.is_target_paper_contribution:
                    continue

                score, asym_score, rationale = self._evaluate_prerequisite(
                    source, target, doc, section_indices
                )

                if score >= self.confidence_threshold:
                    edge = PrerequisiteEdge(
                        source_id=source.concept_id,
                        source_name=source.name,
                        target_id=target.concept_id,
                        target_name=target.name,
                        relation_type="requires_prerequisite",
                        confidence=round(score, 3),
                        asymmetry_score=round(asym_score, 3),
                        epistemic_rationale=rationale,
                    )
                    edges.append(edge)

        return edges

    def _evaluate_prerequisite(
        self,
        source: ConceptEntity,
        target: ConceptEntity,
        doc: DocumentIR,
        sec_indices: Dict[str, int],
    ) -> Tuple[float, float, str]:
        """Evaluate asymmetric score and epistemic rationale for candidate edge source -> target."""
        score_components = []
        rationales = []

        # 1. Check known domain foundational dependencies (Ground-truth epistemic relationships)
        for d_src, d_tgt, d_rat in DOMAIN_DEPENDENCIES:
            if source.name.lower() == d_src.lower() and target.name.lower() == d_tgt.lower():
                return 0.98, 0.95, d_rat

        # 2. Lexical Subsumption (Pan et al. 2019):
        # If source name is a substring/component of target name (e.g. "Oblique Projection" in "Oblique Projection-Based Jamming Cancellation")
        src_clean = source.name.lower()
        tgt_clean = target.name.lower()
        if src_clean != tgt_clean and src_clean in tgt_clean:
            score_components.append(0.92)
            rationales.append(f"'{source.name}' is a foundational component directly subsumed in '{target.name}'.")

        # 3. Taxonomic Hierarchy Delta:
        src_tier = self.taxonomic_tiers.get(source.category, 3.0)
        tgt_tier = self.taxonomic_tiers.get(target.category, 3.0)
        delta_tax = tgt_tier - src_tier

        if delta_tax > 0:
            # Source is in a more foundational category than target (e.g. Math < Methods)
            tax_weight = 0.40 + 0.20 * delta_tax
            score_components.append(min(0.95, tax_weight))
            rationales.append(
                f"Foundational {source.category.value} concept '{source.name}' precedes {target.category.value} concept '{target.name}'."
            )
        elif delta_tax < 0:
            # Source is in a higher category than target (e.g. Method -> Math)
            # Generally penalized unless overriding evidence exists
            score_components.append(0.10)

        # 4. Contribution Bias: Foundational concepts strictly precede novel paper contributions
        if not source.is_target_paper_contribution and target.is_target_paper_contribution:
            score_components.append(0.95)
            rationales.append(
                f"Foundational prerequisite '{source.name}' is essential to construct the paper's novel contribution '{target.name}'."
            )

        # 5. Sequential Offset: Concepts introduced in early sections precede later sections
        src_first_sec = min([sec_indices.get(s, 99) for s in source.source_sections], default=99)
        tgt_first_sec = min([sec_indices.get(s, 99) for s in target.source_sections], default=99)
        delta_pos = tgt_first_sec - src_first_sec

        if delta_pos > 0:
            score_components.append(0.70 + min(0.20, delta_pos * 0.05))
            rationales.append(f"'{source.name}' is introduced prior to '{target.name}' in the document flow.")

        # 6. RefD (Reflected Reference Distance, Manrique et al. 2019)
        # Compute asymmetric co-occurrence across sections
        ref_d = self._compute_ref_d(source, target, doc)
        if ref_d > 0:
            score_components.append(0.60 + 0.35 * ref_d)

        if not score_components:
            return 0.0, 0.0, ""

        final_score = sum(score_components) / len(score_components)
        asymmetry_score = max(0.0, min(1.0, (delta_tax * 0.25) + (ref_d * 0.35) + (0.40 if target.is_target_paper_contribution else 0.0)))

        full_rationale = " ".join(rationales) if rationales else f"Mastery of {source.name} is necessary for understanding {target.name}."

        return final_score, asymmetry_score, full_rationale

    def _compute_ref_d(self, source: ConceptEntity, target: ConceptEntity, doc: DocumentIR) -> float:
        """Calculate Reflected Reference Distance between source and target."""
        # Count mentions of source in target sections vs target in source sections
        src_in_tgt_secs = sum(1 for s in target.source_sections if s in source.source_sections)
        src_count = len(source.source_sections)
        tgt_count = len(target.source_sections)

        if src_count == 0 or tgt_count == 0:
            return 0.0

        r_tgt_to_src = src_in_tgt_secs / tgt_count
        r_src_to_tgt = src_in_tgt_secs / src_count

        max_r = max(r_tgt_to_src, r_src_to_tgt, 0.01)
        # Positive when target relies on source more than source relies on target
        ref_d = (r_tgt_to_src - r_src_to_tgt) / max_r
        return max(-1.0, min(1.0, ref_d))
