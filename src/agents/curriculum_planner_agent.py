"""Stage 5: Curriculum Planner Agent (CurriculumPlannerAgent).

Applies Kahn's topological sort with category priority and enforces strict
DAG Topological Tier Monotonicity: for every edge u -> v, tier(v) >= tier(u).
Organizes prerequisites into 4 pedagogical tiers and generates explainable 'Why Needed' rationales.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Set, Tuple

from src.models.schemas import (
    ConceptCategory,
    ConceptEntity,
    CurriculumConcept,
    CurriculumPlan,
    CurriculumTier,
    PrerequisiteDAG,
    PrerequisiteEdge,
)

TIER_NAMES: Dict[int, str] = {
    1: "Tier 1: Foundational Mathematical & Signal Preliminaries",
    2: "Tier 2: Algorithmic & Communication Foundations",
    3: "Tier 3: Domain Concepts & Theoretical Setting",
    4: "Tier 4: Target Paper Contribution & Innovations",
}

# Priority for Kahn's queue (lower number = higher priority / processed earlier)
CATEGORY_PRIORITY: Dict[ConceptCategory, int] = {
    ConceptCategory.MATHEMATICS: 1,
    ConceptCategory.ALGORITHMS: 2,
    ConceptCategory.THEORIES: 3,
    ConceptCategory.METHODS: 4,
}


class CurriculumPlannerAgent:
    """Agent that plans pedagogical learning sequences enforcing DAG tier monotonicity."""

    def __init__(self) -> None:
        pass

    def plan_curriculum(self, dag: PrerequisiteDAG, paper_title: str) -> CurriculumPlan:
        """Generate pedagogical curriculum plan from PrerequisiteDAG."""
        if not dag.nodes:
            return CurriculumPlan(
                paper_id=dag.paper_id,
                paper_title=paper_title,
                curriculum=[CurriculumTier(tier=t, tier_name=TIER_NAMES[t], concepts=[]) for t in range(1, 5)],
                total_prerequisites=0,
            )

        # 1. Kahn's Topological Sort with Category Priority
        sorted_nodes = self._kahns_topological_sort(dag)

        # 2. Build incoming and outgoing edge maps
        in_edges: Dict[str, List[PrerequisiteEdge]] = {n.concept_id: [] for n in dag.nodes}
        out_edges: Dict[str, List[PrerequisiteEdge]] = {n.concept_id: [] for n in dag.nodes}
        for e in dag.edges:
            in_edges[e.target_id].append(e)
            out_edges[e.source_id].append(e)

        # 3. Assign Tiers enforcing DAG Topological Tier Monotonicity: tier(v) >= tier(u)
        node_tiers = self._assign_monotonic_tiers(sorted_nodes, in_edges)

        # 4. Construct CurriculumConcept objects with explainable "Why Needed" rationales
        node_lookup = {n.concept_id: n for n in dag.nodes}
        tier_buckets: Dict[int, List[CurriculumConcept]] = {1: [], 2: [], 3: [], 4: []}

        for node in sorted_nodes:
            t = node_tiers[node.concept_id]
            direct_prereqs = [e.source_name for e in in_edges[node.concept_id]]
            why_needed = self._synthesize_why_needed(node, out_edges[node.concept_id])

            c_concept = CurriculumConcept(
                concept_id=node.concept_id,
                name=node.name,
                category=node.category.value if isinstance(node.category, ConceptCategory) else str(node.category),
                confidence=node.confidence,
                why_needed=why_needed,
                prerequisites=direct_prereqs,
                is_target_paper_contribution=node.is_target_paper_contribution,
                cso_topic_id=node.cso_topic_id,
                cso_uri=node.cso_uri,
            )
            tier_buckets[t].append(c_concept)

        # 5. Build CurriculumPlan
        tiers = [
            CurriculumTier(tier=t, tier_name=TIER_NAMES[t], concepts=tier_buckets[t])
            for t in range(1, 5)
        ]

        total_prereqs = sum(len(tb) for t, tb in tier_buckets.items() if t < 4)

        return CurriculumPlan(
            paper_id=dag.paper_id,
            paper_title=paper_title,
            curriculum=tiers,
            total_prerequisites=total_prereqs,
        )

    def _kahns_topological_sort(self, dag: PrerequisiteDAG) -> List[ConceptEntity]:
        """Kahn's algorithm using priority queue to favor foundational categories."""
        in_degree: Dict[str, int] = {n.concept_id: 0 for n in dag.nodes}
        adj: Dict[str, List[str]] = {n.concept_id: [] for n in dag.nodes}

        for e in dag.edges:
            adj[e.source_id].append(e.target_id)
            in_degree[e.target_id] += 1

        node_map = {n.concept_id: n for n in dag.nodes}

        # Min-heap queue elements: (priority, is_contribution_int, name, concept_id)
        heap: List[Tuple[int, int, str, str]] = []
        for nid, deg in in_degree.items():
            if deg == 0:
                node = node_map[nid]
                prio = CATEGORY_PRIORITY.get(node.category, 5)
                contrib = 1 if node.is_target_paper_contribution else 0
                heapq.heappush(heap, (prio, contrib, node.name, nid))

        result: List[ConceptEntity] = []

        while heap:
            _, _, _, curr_id = heapq.heappop(heap)
            curr_node = node_map[curr_id]
            result.append(curr_node)

            for neighbor_id in adj.get(curr_id, []):
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    neighbor_node = node_map[neighbor_id]
                    prio = CATEGORY_PRIORITY.get(neighbor_node.category, 5)
                    contrib = 1 if neighbor_node.is_target_paper_contribution else 0
                    heapq.heappush(heap, (prio, contrib, neighbor_node.name, neighbor_id))

        # If any node remains unvisited due to isolated components, append them cleanly
        if len(result) < len(dag.nodes):
            visited_ids = {n.concept_id for n in result}
            for n in dag.nodes:
                if n.concept_id not in visited_ids:
                    result.append(n)

        return result

    def _assign_monotonic_tiers(
        self,
        sorted_nodes: List[ConceptEntity],
        in_edges: Dict[str, List[PrerequisiteEdge]],
    ) -> Dict[str, int]:
        """Assign tiers strictly enforcing DAG Topological Tier Monotonicity: tier(v) >= tier(u)."""
        assigned_tiers: Dict[str, int] = {}

        for node in sorted_nodes:
            # 1. Determine natural base tier
            if node.is_target_paper_contribution:
                base_tier = 4
            elif node.category == ConceptCategory.MATHEMATICS:
                base_tier = 1
            elif node.category == ConceptCategory.ALGORITHMS:
                base_tier = 2
            elif node.category == ConceptCategory.THEORIES:
                base_tier = 3
            else:  # Methods
                base_tier = 3

            # 2. Strict Monotonicity: tier(node) >= max(tier(u)) for all incoming u -> node
            prereq_tiers = [assigned_tiers[e.source_id] for e in in_edges[node.concept_id] if e.source_id in assigned_tiers]
            max_prereq_tier = max(prereq_tiers, default=1)

            effective_tier = max(base_tier, max_prereq_tier)

            # Cap non-contributions at Tier 3 unless forced to 4
            if not node.is_target_paper_contribution and effective_tier > 3:
                effective_tier = 3

            # Contributions must always be in Tier 4
            if node.is_target_paper_contribution:
                effective_tier = 4

            assigned_tiers[node.concept_id] = effective_tier

        # Final verification pass: guarantee 100% monotonicity
        for node in sorted_nodes:
            for e in in_edges[node.concept_id]:
                if e.source_id in assigned_tiers:
                    src_tier = assigned_tiers[e.source_id]
                    tgt_tier = assigned_tiers[node.concept_id]
                    if tgt_tier < src_tier:
                        assigned_tiers[node.concept_id] = src_tier

        return assigned_tiers

    def _synthesize_why_needed(
        self, node: ConceptEntity, outgoing_edges: List[PrerequisiteEdge]
    ) -> str:
        """Synthesize explainable 'Why Needed' rationale based on outgoing dependencies."""
        if node.is_target_paper_contribution:
            return "Primary novel contribution of the paper integrating all preceding mathematical and theoretical foundations."

        if not outgoing_edges:
            return f"Foundational {node.category.value} knowledge necessary to understand the problem domain and paper context."

        dependents = [e.target_name for e in outgoing_edges]
        dep_str = ", ".join(f"'{d}'" for d in dependents[:3])
        if len(dependents) > 3:
            dep_str += f", and {len(dependents) - 3} other downstream concepts"

        # Check if any edge has an epistemic rationale
        specific_rat = ""
        for e in outgoing_edges:
            if e.epistemic_rationale:
                specific_rat = f" {e.epistemic_rationale}"
                break

        return f"Required prerequisite to formulate and comprehend {dep_str}.{specific_rat}"
