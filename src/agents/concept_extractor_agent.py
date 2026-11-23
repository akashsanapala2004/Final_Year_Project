"""Stage 2: Concept Extractor Agent (ConceptExtractorAgent).

Performs 4-way taxonomy concept extraction (Mathematics, Algorithms, Theories, Methods).
Extracts mathematical operators from formulas, mines dynamic scientific terms/acronyms,
and identifies target paper contributions vs external foundational prerequisites.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from src.models.schemas import ConceptCategory, ConceptEntity, DocumentIR, Formula, Section
from src.utils.cso_classifier import get_cso_ontology

# Standardized Taxonomy Knowledge Base
CANONICAL_TAXONOMY: Dict[str, Tuple[ConceptCategory, List[str], str]] = {
    # Mathematics
    "Linear Algebra": (
        ConceptCategory.MATHEMATICS,
        ["linear algebraic", "matrix theory", "vector space"],
        "Foundational vector space and matrix operations.",
    ),
    "Subspaces": (
        ConceptCategory.MATHEMATICS,
        ["subspace", "signal subspace", "interference subspace", "null space"],
        "Linear vector subspaces spanning signal and interference components.",
    ),
    "Oblique Projection": (
        ConceptCategory.MATHEMATICS,
        ["oblique projector", "oblique projection operator", "E_sr"],
        "Non-orthogonal projection operator along a subspace onto another.",
    ),
    "Orthogonal Projection": (
        ConceptCategory.MATHEMATICS,
        ["orthogonal projector", "orthogonal projection operator", "P_A"],
        "Orthogonal subspace projection operator.",
    ),
    "Eigen-Decomposition": (
        ConceptCategory.MATHEMATICS,
        ["eigen decomposition", "eigenvalue decomposition", "spectral decomposition", "eigenvalues", "eigenvectors"],
        "Decomposition of diagonalizable matrices into eigenvalues and eigenvectors.",
    ),
    "Covariance Matrices": (
        ConceptCategory.MATHEMATICS,
        ["covariance matrix", "sample covariance", "correlation matrix", "R_xx", "R_nn"],
        "Statistical matrix representing pairwise covariances across multivariate signals.",
    ),
    "Hypothesis Testing": (
        ConceptCategory.MATHEMATICS,
        ["binary hypothesis testing", "null hypothesis", "alternative hypothesis", "H0", "H1", "decision rule"],
        "Statistical framework for choosing between candidate hypotheses.",
    ),
    "Kronecker Products": (
        ConceptCategory.MATHEMATICS,
        ["kronecker product", "matrix direct product", "\\otimes"],
        "Bilinear matrix operation producing block matrix representations.",
    ),
    "Pseudo-Inverse": (
        ConceptCategory.MATHEMATICS,
        ["moore-penrose inverse", "pseudoinverse", "generalized inverse", "A^+", "A^\\dagger"],
        "Generalized matrix inverse for non-square or singular matrices.",
    ),
    "Gaussian Processes": (
        ConceptCategory.MATHEMATICS,
        ["gaussian process", "gp regression", "kernel distribution"],
        "Non-parametric Bayesian prior over continuous functions.",
    ),
    "Convex Optimization": (
        ConceptCategory.MATHEMATICS,
        ["convex program", "convex problem", "lagrangian duality", "kkt conditions"],
        "Mathematical optimization over convex sets with convex objective functions.",
    ),
    "Graph Theory": (
        ConceptCategory.MATHEMATICS,
        ["graph theoretical", "undirected graph", "directed graph", "adjacency matrix"],
        "Mathematical study of vertices, edges, and relational networks.",
    ),
    "Bayesian Probability": (
        ConceptCategory.MATHEMATICS,
        ["bayesian inference", "bayes rule", "posterior probability", "prior distribution"],
        "Probabilistic reasoning updating hypothesis priors with observed evidence.",
    ),
    "Boolean Logic": (
        ConceptCategory.MATHEMATICS,
        ["boolean algebra", "propositional logic", "sat solver", "boolean satisfiability"],
        "Algebra of truth values and logical connective operators.",
    ),
    "Fourier Transform": (
        ConceptCategory.MATHEMATICS,
        ["fourier analysis", "discrete fourier transform", "frequency domain transform", "fourier matrix"],
        "Mathematical transform representing signals in the complex frequency domain.",
    ),
    "Singular Value Decomposition": (
        ConceptCategory.MATHEMATICS,
        ["svd", "singular values", "singular vectors"],
        "Factorization of real or complex matrices into singular vectors.",
    ),

    # Algorithms
    "GLRT": (
        ConceptCategory.ALGORITHMS,
        ["generalized likelihood ratio test", "generalized likelihood ratio", "glr test"],
        "Algorithmic decision statistic replacing unknown parameters with maximum likelihood estimates.",
    ),
    "IFFT": (
        ConceptCategory.ALGORITHMS,
        ["inverse fast fourier transform", "ifft algorithm", "inverse fft"],
        "Efficient algorithmic implementation of the inverse discrete Fourier transform.",
    ),
    "FFT": (
        ConceptCategory.ALGORITHMS,
        ["fast fourier transform", "fft algorithm", "cooley-tukey"],
        "Algorithmic implementation computing discrete Fourier transforms in O(N log N).",
    ),
    "Graham's Algorithm": (
        ConceptCategory.ALGORITHMS,
        ["graham algorithm", "graham scan", "convex hull algorithm"],
        "Algorithmic procedure for computing the convex hull of finite 2D points.",
    ),
    "Hypertree Decomposition": (
        ConceptCategory.ALGORITHMS,
        ["hypertree width", "tree decomposition algorithm", "generalized hypertree decomposition"],
        "Structural graph algorithm decomposing hypergraphs into acyclic tree structures.",
    ),
    "Gradient Descent": (
        ConceptCategory.ALGORITHMS,
        ["steepest descent", "stochastic gradient descent", "gradient optimization"],
        "Iterative first-order optimization algorithm for finding function minima.",
    ),
    "Genetic Algorithms": (
        ConceptCategory.ALGORITHMS,
        ["genetic algorithm", "evolutionary algorithm", "genetic search"],
        "Metaheuristic optimization algorithm inspired by natural selection.",
    ),
    "Dynamic Programming": (
        ConceptCategory.ALGORITHMS,
        ["dp algorithm", "bellman equation", "memoization"],
        "Algorithmic paradigm breaking complex problems into overlapping subproblems.",
    ),
    "Graph Traversal": (
        ConceptCategory.ALGORITHMS,
        ["depth-first search", "dfs", "breadth-first search", "bfs"],
        "Algorithms for visiting all vertices in a graph structure.",
    ),
    "Clustering and Classification": (
        ConceptCategory.ALGORITHMS,
        ["k-means clustering", "svm classification", "nearest neighbor", "spectral clustering"],
        "Algorithmic partitioning and supervised categorisation of high-dimensional data.",
    ),

    # Theories
    "Cognitive Radio": (
        ConceptCategory.THEORIES,
        ["cognitive radio networks", "opportunistic spectrum access", "secondary users", "crn"],
        "Theoretical paradigm for intelligent wireless systems dynamically utilizing spectrum.",
    ),
    "Physical Layer Security": (
        ConceptCategory.THEORIES,
        ["secrecy capacity", "wiretap channel", "information-theoretic security", "pls"],
        "Theory exploiting wireless channel noise and fading to secure data transmission.",
    ),
    "Spectrum Sensing": (
        ConceptCategory.THEORIES,
        ["spectrum detection", "sensing model", "primary user detection", "channel vacancy"],
        "Theoretical framework for detecting idle frequency channels in cognitive radio.",
    ),
    "Computational Complexity": (
        ConceptCategory.THEORIES,
        ["np-complete", "np-hard", "complexity theory", "polynomial-time reduction"],
        "Theory classifying computational problems by resource consumption bounds.",
    ),
    "Game Theory": (
        ConceptCategory.THEORIES,
        ["nash equilibrium", "non-cooperative game", "utility function", "strategic game"],
        "Mathematical theory of strategic interactions among rational decision makers.",
    ),
    "Deterministic Replay": (
        ConceptCategory.THEORIES,
        ["record-and-replay", "replay debugging", "execution replay"],
        "System theory guaranteeing reproducible execution of concurrent program runs.",
    ),
    "Domain-Specific Languages and Type Theory": (
        ConceptCategory.THEORIES,
        ["type theory", "dsl", "static typing", "operational semantics", "lambda calculus"],
        "Formal linguistic theory for syntax and static semantic safety guarantees.",
    ),
    "Information Theory": (
        ConceptCategory.THEORIES,
        ["shannon capacity", "mutual information", "entropy", "channel capacity"],
        "Mathematical theory quantifying information storage and transmission limits.",
    ),

    # Methods
    "OFDM": (
        ConceptCategory.METHODS,
        ["orthogonal frequency-division multiplexing", "ofdm transmission", "cyclic prefix ofdm"],
        "Multicarrier modulation method dividing wideband channels into orthogonal subcarriers.",
    ),
    "Access Control": (
        ConceptCategory.METHODS,
        ["access control mechanism", "role-based access control", "rbac", "attribute-based access"],
        "Security methodology regulating authorized interactions with system resources.",
    ),
    "Jamming Cancellation": (
        ConceptCategory.METHODS,
        ["anti-jamming", "jamming suppression", "interference mitigation", "jammer elimination"],
        "Signal processing method nulling intentional interfering transmissions.",
    ),
    "Stemming": (
        ConceptCategory.METHODS,
        ["porter stemmer", "morphological stemming", "word stemming"],
        "Text processing method reducing derived words to their linguistic base stem.",
    ),
    "Hypergraph Reduction": (
        ConceptCategory.METHODS,
        ["hypergraph simplification", "reduced hypergraph", "hyperedge contraction"],
        "Graph method pruning hypergraphs while preserving structural connectivity.",
    ),
    "Case-Based Reasoning": (
        ConceptCategory.METHODS,
        ["cbr", "case retrieval", "case adaptation", "analogical reasoning"],
        "Method solving novel problems by retrieving and adapting past case solutions.",
    ),
    "Model Checking and Formal Verification": (
        ConceptCategory.METHODS,
        ["formal verification", "model checking", "temporal logic verification", "ltl checking"],
        "Automated methodology formally verifying state models against specifications.",
    ),
}

# Formula Operator Patterns
FORMULA_OPERATOR_SIGNATURES: List[Tuple[re.Pattern, str, ConceptCategory]] = [
    (re.compile(r"(E_\{?[sS][rR]\}?|P_\{?[a-zA-Z]+\}?|\\text\{proj\}|projection)", re.IGNORECASE), "Oblique Projection", ConceptCategory.MATHEMATICS),
    (re.compile(r"(\\Sigma|R_\{?[a-zA-Z0-9]+\}?|E\[[a-zA-Z0-9\^\\*]+\]|covariance)", re.IGNORECASE), "Covariance Matrices", ConceptCategory.MATHEMATICS),
    (re.compile(r"(\\Lambda\(|L\(|H_0|H_1|\\arg\\max|likelihood\s+ratio)", re.IGNORECASE), "Hypothesis Testing", ConceptCategory.MATHEMATICS),
    (re.compile(r"(\\otimes|kronecker)", re.IGNORECASE), "Kronecker Products", ConceptCategory.MATHEMATICS),
    (re.compile(r"(\^\{\+\}|\^\{\\dagger\}|pseudoinverse)", re.IGNORECASE), "Pseudo-Inverse", ConceptCategory.MATHEMATICS),
    (re.compile(r"(\\min|\\max|\\arg\\min|\\arg\\max|\\nabla|s\.t\.|subject\s+to)", re.IGNORECASE), "Convex Optimization", ConceptCategory.MATHEMATICS),
    (re.compile(r"(F\^H|F_N|e\^\{-j2\\pi|discrete\s+fourier)", re.IGNORECASE), "Fourier Transform", ConceptCategory.MATHEMATICS),
    (re.compile(r"(\\lambda_\{?[0-9i]+\}?|eigenvalue|eigenvector)", re.IGNORECASE), "Eigen-Decomposition", ConceptCategory.MATHEMATICS),
    (re.compile(r"(G\s*=\s*\([VE],\s*[VE]\)|\\mathcal\{H\}\s*=\s*\(V,\s*E\))", re.IGNORECASE), "Graph Theory", ConceptCategory.MATHEMATICS),
]

# Stop phrases to ignore when dynamically mining capitalized phrases
DYNAMIC_STOP_WORDS = {
    "The Paper", "In This", "This Paper", "We Propose", "We Present", "Figure 1",
    "Figure 2", "Figure 3", "Table 1", "Table 2", "Section 1", "Section 2",
    "Department", "University", "IEEE", "ACM", "Springer", "Elsevier",
    "Conference", "Proceedings", "Abstract", "Introduction", "Related Work",
    "Conclusion", "Methodology", "Performance Evaluation", "Experimental Results",
    "Simulation Results", "Future Work", "Our Method", "Our Proposed", "System Model",
    "Proposed Methodology", "Problem Formulation", "Signal Model", "Mathematical Model",
    "Experimental Setup", "Performance Comparison", "Prior Work",
}


class ConceptExtractorAgent:
    """Agent responsible for 4-way taxonomy concept extraction from DocumentIR."""

    def __init__(self) -> None:
        self.taxonomy = CANONICAL_TAXONOMY
        self.cso = get_cso_ontology()

    def extract_concepts(self, doc: DocumentIR) -> List[ConceptEntity]:
        """Extract structured scientific concepts from paper document IR."""
        extracted: Dict[str, ConceptEntity] = {}
        all_text_sections: List[Tuple[str, str]] = []

        # Collect text from abstract and sections
        if doc.abstract:
            all_text_sections.append(("sec_abstract", doc.abstract))
        for sec in doc.sections:
            combined_sec_text = sec.heading + ". " + " ".join(sec.paragraphs)
            all_text_sections.append((sec.section_id, combined_sec_text))

        # 1. Match Canonical Taxonomy Concepts
        for concept_name, (cat, aliases, _) in self.taxonomy.items():
            matched_sections: List[str] = []
            found_aliases: List[str] = []

            # Check primary name
            pat_primary = re.compile(rf"\b{re.escape(concept_name)}\b", re.IGNORECASE)
            for sec_id, text in all_text_sections:
                if pat_primary.search(text):
                    if sec_id not in matched_sections:
                        matched_sections.append(sec_id)

            # Check aliases
            for alias in aliases:
                pat_alias = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
                for sec_id, text in all_text_sections:
                    if pat_alias.search(text):
                        if sec_id not in matched_sections:
                            matched_sections.append(sec_id)
                        if alias not in found_aliases:
                            found_aliases.append(alias)

            if matched_sections:
                cid = self._to_concept_id(concept_name)
                # Foundational math and broad domain theories are NEVER paper contributions
                is_contrib = False
                extracted[concept_name] = ConceptEntity(
                    concept_id=cid,
                    name=concept_name,
                    category=cat,
                    confidence=0.98 if len(matched_sections) > 1 else 0.88,
                    source_sections=matched_sections,
                    aliases=found_aliases,
                    is_target_paper_contribution=is_contrib,
                )

        # 2. Extract Mathematical Operators from Formulas (<tei:formula>)
        for sec in doc.sections:
            for formula in sec.formulas:
                formula_combined = f"{formula.raw_text} {formula.latex}"
                for pattern, concept_name, cat in FORMULA_OPERATOR_SIGNATURES:
                    if pattern.search(formula_combined):
                        cid = self._to_concept_id(concept_name)
                        if concept_name in extracted:
                            ent = extracted[concept_name]
                            if sec.section_id not in ent.source_sections:
                                ent.source_sections.append(sec.section_id)
                        else:
                            extracted[concept_name] = ConceptEntity(
                                concept_id=cid,
                                name=concept_name,
                                category=cat,
                                confidence=0.95,
                                source_sections=[sec.section_id],
                                aliases=[f"formula:{formula.formula_id}"],
                                is_target_paper_contribution=False,
                            )

        # 3. Dynamic Concept Mining: Pattern (ACRONYM) and Multi-Word Capitalized Scientific Phrases
        mined_concepts = self._mine_dynamic_concepts(doc, all_text_sections)
        for m_name, m_cat, m_secs, m_aliases, is_contrib in mined_concepts:
            cid = self._to_concept_id(m_name)
            if m_name in extracted:
                ent = extracted[m_name]
                for s in m_secs:
                    if s not in ent.source_sections:
                        ent.source_sections.append(s)
                for a in m_aliases:
                    if a not in ent.aliases:
                        ent.aliases.append(a)
                if is_contrib and ent.category not in (ConceptCategory.MATHEMATICS, ConceptCategory.THEORIES):
                    ent.is_target_paper_contribution = True
            else:
                extracted[m_name] = ConceptEntity(
                    concept_id=cid,
                    name=m_name,
                    category=m_cat,
                    confidence=0.85,
                    source_sections=m_secs,
                    aliases=m_aliases,
                    is_target_paper_contribution=is_contrib,
                )

        # 4. Ontology Identification and Classification via CSO.3.3.owl
        for ent in list(extracted.values()):
            cso_topic = self.cso.identify(ent.name)
            if not cso_topic and ent.aliases:
                for a in ent.aliases:
                    cso_topic = self.cso.identify(a)
                    if cso_topic:
                        break
            if cso_topic:
                ent.cso_topic_id = cso_topic.topic_id
                ent.cso_uri = cso_topic.uri
                for eq in cso_topic.equivalents:
                    eq_clean = eq.replace("_", " ").title()
                    if eq_clean not in ent.aliases:
                        ent.aliases.append(eq_clean)

        # 5. Final Contribution Disambiguation based on Title and Abstract
        self._disambiguate_contributions(doc, extracted)

        return list(extracted.values())

    def _to_concept_id(self, name: str) -> str:
        """Convert concept name to clean snake_case ID."""
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
        return f"c_{clean}"

    def _mine_dynamic_concepts(
        self, doc: DocumentIR, all_text_sections: List[Tuple[str, str]]
    ) -> List[Tuple[str, ConceptCategory, List[str], List[str], bool]]:
        """Mine multi-word scientific concepts and acronym definitions like Pattern (ACRONYM)."""
        results: List[Tuple[str, ConceptCategory, List[str], List[str], bool]] = []
        seen_names: Set[str] = set()

        # Acronym regex: "Case-Based Reasoning (CBR)"
        acronym_pattern = re.compile(r"\b([A-Z][a-zA-Z\s\-]{3,40})\s*\(([A-Z0-9]{2,8})\)")
        for sec_id, text in all_text_sections:
            for match in acronym_pattern.finditer(text):
                full_name = " ".join(match.group(1).split())
                acronym = match.group(2).strip()
                if full_name in DYNAMIC_STOP_WORDS or len(full_name.split()) < 2:
                    continue
                if full_name.lower() in seen_names:
                    continue

                seen_names.add(full_name.lower())
                cat = self._infer_concept_category(full_name)
                # Check if this matches paper contribution pattern
                is_contrib = self._is_potential_paper_contribution(full_name, doc)
                results.append((full_name, cat, [sec_id], [acronym], is_contrib))

        # Scientific multi-word phrases from titles and headings
        heading_texts = [sec.heading for sec in doc.sections]
        if doc.title:
            heading_texts.insert(0, doc.title)

        multi_word_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
        for h in heading_texts:
            for match in multi_word_pattern.finditer(h):
                phrase = match.group(1).strip()
                words = phrase.split()
                if phrase in DYNAMIC_STOP_WORDS or len(words) < 2:
                    continue
                if words[0] in {"Based", "Using", "Via", "And", "With", "For", "In", "On", "Of", "Over", "Under"}:
                    continue
                if phrase.lower() in seen_names:
                    continue

                seen_names.add(phrase.lower())
                cat = self._infer_concept_category(phrase)
                is_contrib = self._is_potential_paper_contribution(phrase, doc)
                results.append((phrase, cat, ["sec_title_or_head"], [], is_contrib))

        return results

    def _infer_concept_category(self, phrase: str) -> ConceptCategory:
        """Infer 4-way category for dynamically mined concept using CSO ontology and rules."""
        # 1. Ontology-guided classification from CSO.3.3.owl
        cso_topic = self.cso.identify(phrase)
        if cso_topic:
            return cso_topic.category

        p_low = phrase.lower()
        # 1. Methods take precedence for composite systems/schemes
        if any(w in p_low for w in ["cancellation", "reduction", "reasoning", "multiplexing", "modulation", "scheme", "allocation", "technique", "method", "filtering", "estimation", "checking", "control", "access"]):
            return ConceptCategory.METHODS
        # 2. Algorithms
        if any(w in p_low for w in ["algorithm", "traversal", "search", "descent", "sort", "decomposition", "solver", "heuristic", "scan", "test"]):
            return ConceptCategory.ALGORITHMS
        # 3. Mathematics
        if any(w in p_low for w in ["algebra", "subspace", "matrix", "vector", "distribution", "calculus", "norm", "integral", "equation", "projection", "eigen", "space"]):
            return ConceptCategory.MATHEMATICS
        # 4. Theories
        if any(w in p_low for w in ["theory", "complexity", "principle", "theorem", "law", "game", "property", "security", "framework", "radio", "sensing"]):
            return ConceptCategory.THEORIES
        # Default to Methods
        return ConceptCategory.METHODS

    def _is_potential_paper_contribution(self, phrase: str, doc: DocumentIR) -> bool:
        """Heuristic check if a phrase represents the target paper's novel contribution."""
        p_low = phrase.lower()
        title_low = (doc.title or "").lower()
        abstract_low = (doc.abstract or "").lower()

        # Strict Anti-Pattern: Foundational math and broad domain theories are NEVER paper contributions
        cat = self._infer_concept_category(phrase)
        if cat in (ConceptCategory.MATHEMATICS, ConceptCategory.THEORIES):
            return False

        # If it is in the title, and title indicates proposed method
        if p_low in title_low:
            return True

        # Check contribution phrases in abstract
        contrib_triggers = [
            f"we propose {p_low}",
            f"we present {p_low}",
            f"we introduce {p_low}",
            f"our proposed {p_low}",
            f"novel {p_low}",
            f"new {p_low}",
            f"proposed {p_low}",
        ]
        if any(t in abstract_low for t in contrib_triggers):
            return True

        return False

    def _disambiguate_contributions(self, doc: DocumentIR, extracted: Dict[str, ConceptEntity]) -> None:
        """Enforce strict contribution rules: foundational math/theories cannot be contributions."""
        title_low = (doc.title or "").lower()
        has_any_contrib = any(e.is_target_paper_contribution for e in extracted.values())

        for name, ent in extracted.items():
            # Rule: Math and broad theories are NEVER paper contributions
            if ent.category in (ConceptCategory.MATHEMATICS, ConceptCategory.THEORIES):
                ent.is_target_paper_contribution = False
                continue

            # If method appears in title or proposed phrasing, mark as contribution
            if ent.category in (ConceptCategory.METHODS, ConceptCategory.ALGORITHMS):
                if name.lower() in title_low and ("using" in title_low or "via" in title_low or "based" in title_low):
                    # In "X based on Y" or "X using Y", X is the contribution, Y is the prerequisite
                    parts = re.split(r"\b(?:using|via|based on|for)\b", title_low)
                    if parts and name.lower() in parts[0]:
                        ent.is_target_paper_contribution = True

        # If no contribution was marked, find the primary Method or Algorithm in the title
        if not any(e.is_target_paper_contribution for e in extracted.values()):
            for name, ent in extracted.items():
                if ent.category in (ConceptCategory.METHODS, ConceptCategory.ALGORITHMS) and name.lower() in title_low:
                    ent.is_target_paper_contribution = True
                    break
