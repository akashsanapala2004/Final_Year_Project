"""Stage 4: Graph Reasoning Agent (GraphReasoningAgent).

Constructs the Prerequisite Knowledge Graph, applies Greedy Feedback Arc Set (FAS)
via DFS cycle detection to ensure strict DAG acyclicity, performs Transitive Reduction
to isolate the essential prerequisite backbone, and computes graph topological metrics.
Pure Python standard library implementation with zero external dependencies.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from src.models.schemas import (
    ConceptEntity,
    GraphMetrics,
    PrerequisiteDAG,
    PrerequisiteEdge,
)


class GraphReasoningAgent:
    """Agent that builds DAG, resolves cycles via Greedy FAS, and computes Transitive Reduction."""

    def __init__(self, enable_transitive_reduction: bool = True) -> None:
        self.enable_transitive_reduction = enable_transitive_reduction

    def build_dag(
        self, paper_id: str, concepts: List[ConceptEntity], edges: List[PrerequisiteEdge]
    ) -> PrerequisiteDAG:
        """Construct a verified Prerequisite DAG from concepts and candidate edges."""
        node_map: Dict[str, ConceptEntity] = {c.concept_id: c for c in concepts}
        edge_map: Dict[Tuple[str, str], PrerequisiteEdge] = {}

        for e in edges:
            if e.source_id in node_map and e.target_id in node_map and e.source_id != e.target_id:
                # Keep edge with highest confidence if duplicate
                key = (e.source_id, e.target_id)
                if key not in edge_map or e.confidence > edge_map[key].confidence:
                    edge_map[key] = e

        # 1. Greedy Feedback Arc Set (FAS) cycle elimination
        active_edges, cycles_resolved = self._greedy_feedback_arc_set(
            list(node_map.keys()), list(edge_map.values())
        )

        # 2. Transitive Reduction: remove redundant bypass edges (u -> v when u -> ... -> v exists)
        if self.enable_transitive_reduction:
            reduced_edges = self._transitive_reduction(list(node_map.keys()), active_edges)
        else:
            reduced_edges = active_edges

        # 3. Build final adjacency list
        adjacency_list: Dict[str, List[str]] = {nid: [] for nid in node_map}
        for e in reduced_edges:
            adjacency_list[e.source_id].append(e.target_id)

        # Sort adjacency list for determinism
        for nid in adjacency_list:
            adjacency_list[nid].sort()

        # 4. Compute Graph Metrics
        num_nodes = len(node_map)
        num_edges = len(reduced_edges)
        max_possible_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
        density = round(num_edges / max_possible_edges, 4) if num_nodes > 1 else 0.0

        is_dag = self._is_acyclic(list(node_map.keys()), adjacency_list)

        metrics = GraphMetrics(
            num_nodes=num_nodes,
            num_edges=num_edges,
            is_dag=is_dag,
            cycles_detected_and_resolved=cycles_resolved,
            density=density,
        )

        return PrerequisiteDAG(
            paper_id=paper_id,
            metrics=metrics,
            nodes=list(node_map.values()),
            edges=reduced_edges,
            adjacency_list=adjacency_list,
        )

    def _greedy_feedback_arc_set(
        self, node_ids: List[str], edges: List[PrerequisiteEdge]
    ) -> Tuple[List[PrerequisiteEdge], int]:
        """Iteratively detect cycles using DFS and remove lowest-confidence edge until acyclic."""
        current_edges = { (e.source_id, e.target_id): e for e in edges }
        cycles_resolved = 0

        while True:
            # Build current adjacency map
            adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}
            for (u, v) in current_edges:
                adj[u].append(v)

            cycle = self._find_cycle_dfs(node_ids, adj)
            if not cycle:
                # Graph is strictly acyclic
                break

            # Identify cycle edges
            cycle_edges: List[Tuple[str, str]] = []
            for i in range(len(cycle) - 1):
                cycle_edges.append((cycle[i], cycle[i + 1]))

            # Find edge with lowest confidence in the cycle
            min_edge_key = min(
                cycle_edges,
                key=lambda k: (
                    current_edges[k].confidence,
                    current_edges[k].asymmetry_score,
                ),
            )

            # Remove lowest confidence edge
            del current_edges[min_edge_key]
            cycles_resolved += 1

        return list(current_edges.values()), cycles_resolved

    def _find_cycle_dfs(self, node_ids: List[str], adj: Dict[str, List[str]]) -> Optional[List[str]]:
        """Find a directed cycle using 3-color DFS (0: White/Unvisited, 1: Gray/Visiting, 2: Black/Visited)."""
        color: Dict[str, int] = {nid: 0 for nid in node_ids}
        parent: Dict[str, Optional[str]] = {nid: None for nid in node_ids}

        for start_node in node_ids:
            if color[start_node] != 0:
                continue

            # Stack for DFS with explicit traversal path
            call_stack: List[Tuple[str, int]] = [(start_node, 0)]
            color[start_node] = 1
            path: List[str] = [start_node]

            while call_stack:
                u, child_idx = call_stack[-1]
                neighbors = adj.get(u, [])

                if child_idx < len(neighbors):
                    v = neighbors[child_idx]
                    # Update next index for u
                    call_stack[-1] = (u, child_idx + 1)

                    if color[v] == 1:
                        # Found cycle: v is in recursion stack
                        cycle_start_idx = path.index(v)
                        return path[cycle_start_idx:] + [v]
                    elif color[v] == 0:
                        color[v] = 1
                        parent[v] = u
                        path.append(v)
                        call_stack.append((v, 0))
                else:
                    color[u] = 2
                    path.pop()
                    call_stack.pop()

        return None

    def _transitive_reduction(
        self, node_ids: List[str], edges: List[PrerequisiteEdge]
    ) -> List[PrerequisiteEdge]:
        """Remove redundant bypass edges (u -> v) when an alternate path of length >= 2 exists."""
        edge_map = { (e.source_id, e.target_id): e for e in edges }
        adj: Dict[str, Set[str]] = {nid: set() for nid in node_ids}
        for (u, v) in edge_map:
            adj[u].add(v)

        essential_edges: List[PrerequisiteEdge] = []

        for (u, v), edge in edge_map.items():
            # Temporarily remove direct edge (u -> v)
            adj[u].remove(v)

            # Check if v is still reachable from u via other paths
            if not self._is_reachable(u, v, adj):
                # Essential edge, keep it
                essential_edges.append(edge)
                adj[u].add(v)
            else:
                # Redundant bypass edge! Do not restore into essential graph
                pass

        return essential_edges

    def _is_reachable(self, start: str, target: str, adj: Dict[str, Set[str]]) -> bool:
        """BFS reachability check from start to target."""
        visited: Set[str] = {start}
        queue = deque([start])

        while queue:
            curr = queue.popleft()
            for neighbor in adj.get(curr, set()):
                if neighbor == target:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def _is_acyclic(self, node_ids: List[str], adj: Dict[str, List[str]]) -> bool:
        """Verify strict DAG acyclicity."""
        return self._find_cycle_dfs(node_ids, adj) is None
