# Master Rebuild Prompt: Scientific Prerequisite Concept Extraction & Knowledge Graph System

Copy and paste the exact prompt block below into any AI agent, LLM, or pass it to a software engineer to reconstruct the entire multi-agent system from scratch with zero mistakes.

```markdown
You are an expert AI Research Architect and Senior Systems Engineer.
Your task is to build a production-grade, modular Multi-Agent System from scratch in Python that solves the following academic problem:

### Core Problem Statement
"Given an academic research paper (PDF or GROBID TEI-XML), automatically determine the foundational prerequisite concepts (Mathematics, Algorithms, Theories, Methods) required before reading the paper, construct a Prerequisite Knowledge Graph (Directed Acyclic Graph), generate an optimal topological learning sequence (pedagogical curriculum), and ground every prerequisite in verifiable textual evidence and mathematical formulas from the paper."

### Theoretical Foundations
Directly operationalize the algorithms from two seminal research works:
1. "Exploring knowledge graphs for the identification of concept prerequisites" (Manrique, Nunes, & Mariño, 2019):
   - Reflected Reference Distance (RefD) for asymmetric directional prerequisite learning.
   - Graph topological properties (centrality, reachability, DAG acyclicity).
2. "Concept Extraction and Prerequisite Relation Learning from Educational Data" (Pan et al., AAAI 2019 / iPRL framework):
   - Hybrid prerequisite scoring combining feature classification (taxonomic hierarchy delta, sequential position offset, lexical subsumption) with global graph recovery constraints.

---

### CRITICAL ANTI-PATTERNS (Do NOT Make These Mistakes)
1. DO NOT use generic spaCy `en_core_web_sm` for entity extraction: It extracts people, single letters ('H', 'K'), dates, and table numbers instead of scientific concepts.
2. DO NOT perform pairwise O(N^2) character slicing between entities: This creates hundreds of megabytes of noisy strings ("is", "are", "consists of") that lack prerequisite directionality.
3. DO NOT ignore `<tei:formula>` mathematical equation tags in GROBID XMLs: Mathematical formulas contain crucial foundational operators (e.g. projection operators, Fourier matrices, covariance matrices).
4. DO NOT accept copyright disclaimers as paper titles: GROBID often puts ACM/IEEE copyright text ("Permission to make digital...") into the title element. You must reject copyright boilerplate.
5. DO NOT violate DAG topological tier monotonicity: In the learning curriculum, if concept A is a prerequisite for concept B (A -> B), tier(B) MUST be >= tier(A). A student must never be told to learn Tier 3 before its prerequisite in Tier 2.
6. DO NOT use fragile external heavy dependencies: Build the core graph algorithms (DFS cycle detection, Greedy Feedback Arc Set, transitive reduction, Kahn's topological sort) using pure Python standard library (`collections`, `dataclasses`, `xml.etree.ElementTree`, `re`, `pathlib`, `argparse`, `concurrent.futures`) so the system runs with zero installation fragility.

---

### REQUIRED DIRECTORY STRUCTURE
Rebuild the project matching this exact layout:
```text
final_yr_project/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── paper_parser_agent.py          # Stage 1: Document & Formula parsing
│   │   ├── concept_extractor_agent.py     # Stage 2: 4-Way taxonomy concept extraction
│   │   ├── prerequisite_relation_agent.py # Stage 3: Asymmetric RefD & iPRL relation learning
│   │   ├── graph_reasoning_agent.py       # Stage 4: DAG construction & FAS cycle removal
│   │   ├── curriculum_planner_agent.py    # Stage 5: Topological sort & 4-tier curriculum
│   │   ├── evidence_grounding_agent.py    # Stage 6: Grounded in-text quotes & citations
│   │   └── orchestrator.py                # Pipeline coordinator & CLI
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py                     # Dataclasses & Pydantic models
│   │   └── train_prereq_classifier.py     # Pairwise classifier trainer
│   └── utils/
│       ├── __init__.py
│       └── graph_viz.py                   # Mermaid flowchart exporter
├── tests/
│   ├── __init__.py
│   ├── test_schemas.py
│   ├── test_agents.py
│   └── test_tools_and_cli.py
├── data/
│   ├── raw_pdfs/                          # Input PDFs
│   └── grobid_output/                     # Extracted TEI-XMLs
├── output/                                # Generated curricula and graphs
├── run.py                                 # Single-paper and list runner CLI
├── query.py                               # Category and concept search CLI
├── batch_train.py                         # Parallel 1000-PDF multi-core runner
├── run_tests.py                           # Master test suite runner
├── .gitignore
├── requirements.txt
└── README.md
```

---

### DETAILED AGENT SPECIFICATIONS

#### 1. Data Models (`src/models/schemas.py`)
Implement typed Python dataclasses supporting `to_dict()` and `to_json()`:
- `ConceptCategory`: Enum (`"Mathematics"`, `"Algorithms"`, `"Theories"`, `"Methods"`).
- `Formula`: `formula_id`, `raw_text`, `latex`, `section_id`.
- `Section`: `section_id`, `heading`, `category`, `paragraphs`, `formulas`, `citations`.
- `DocumentIR`: `paper_id`, `title`, `abstract`, `sections`, `preliminaries`, `citations`.
- `ConceptEntity`: `concept_id`, `name`, `category`, `confidence`, `source_sections`, `aliases`, `is_target_paper_contribution`.
- `PrerequisiteEdge`: `source_id`, `source_name`, `target_id`, `target_name`, `relation_type`, `confidence`, `asymmetry_score`, `epistemic_rationale`.
- `GraphMetrics`: `num_nodes`, `num_edges`, `is_dag`, `cycles_detected_and_resolved`, `density`.
- `PrerequisiteDAG`: `paper_id`, `metrics`, `nodes`, `edges`, `adjacency_list`.
- `CurriculumTier`: `tier` (1 to 4), `tier_name`, `concepts` (`CurriculumConcept`).
- `CurriculumPlan`: `paper_id`, `paper_title`, `curriculum` (list of tiers), `total_prerequisites`.
- `EvidenceRecord`: `prerequisite_concept`, `dependent_concept`, `evidence_type`, `section`, `quote`, `formula_reference`, `cited_foundational_paper`.

#### 2. Stage 1: Document & Formula Parser (`PaperParserAgent`)
- Accepts paper ID or XML file path. Parses GROBID TEI-XML using `xml.etree.ElementTree`.
- Rejects copyright boilerplates in title (e.g., regex check for "permission to make digital", "copyright", "is granted without fee") and falls back to first clean section header or paragraph.
- Traverses all `<tei:div>`, extracts headings, paragraphs, and explicitly parses `<tei:formula>` preserving raw text and section association.
- Extracts citation links `<tei:ref type="bibr">` and maps them to `<tei:listBibl>` bibliography metadata.

#### 3. Stage 2: Concept Extraction (`ConceptExtractorAgent`)
- Implements a broad, multi-domain taxonomy across:
  - **Mathematics**: Linear algebra, subspaces, oblique projection, eigen-decomposition, covariance matrices, hypothesis testing, Kronecker products, pseudo-inverse, Gaussian processes, convex optimization, graph theory, Bayesian probability, Boolean logic.
  - **Algorithms**: GLRT, IFFT/FFT, Graham's algorithm, hypertree decomposition, gradient descent, genetic algorithms, dynamic programming, graph traversal (DFS/BFS), clustering/classification.
  - **Theories**: Cognitive radio, physical layer security, spectrum sensing, computational complexity (NP-complete), game theory, deterministic replay, domain-specific languages & type theory.
  - **Methods**: OFDM, access control, jamming cancellation, stemming, hypergraph reduction, case-based reasoning (CBR), model checking & formal verification.
- **Formula Extraction**: Extracts operators directly from `<tei:formula>` blocks and links them to their actual `section_id`.
- **Dynamic Concept Mining**: Extracts multi-word capitalized scientific phrases from headings/paragraphs, and captures acronym definitions `Pattern (ACRONYM)` like `Case-Based Reasoning (CBR)`.
- **Contribution Identifier**: Determines if a concept is the novel contribution of the paper (`is_target_paper_contribution = True`) vs an external prerequisite (`False`). Foundational math and broad domain theories are NEVER marked as paper contributions.

#### 4. Stage 3: Asymmetric Prerequisite Learning (`PrerequisiteRelationAgent`)
- Discovers directed edges ($A \to B$) where mastering $A$ is required to understand $B$.
- Strict rule: A paper's target contribution can NEVER precede foundational external prerequisites.
- Computes asymmetric scoring based on:
  1. Taxonomic Delta: $\text{tier}(\text{Math}) < \text{tier}(\text{Theories}) < \text{tier}(\text{Algorithms}) < \text{tier}(\text{Methods})$.
  2. Contribution Bias: Foundational concepts strictly precede novel paper contributions.
  3. Sequential Offset: Concepts defined in early sections (Preliminaries/Intro) precede those in Methodology.
  4. Lexical Subsumption: If $A$'s name is a component of $B$'s name (e.g. *Oblique Projection* in *Oblique Projection-Based Jamming Cancellation*), $A \to B$.
- Generates clear epistemic rationales explaining why $A$ is needed for $B$.

#### 5. Stage 4: Prerequisite KG & DAG Construction (`GraphReasoningAgent`)
- Builds directed graph $G = (V, E)$.
- **Greedy Feedback Arc Set (FAS)**: Uses DFS-based cycle detection to locate directed cycles ($A \to B \to \dots \to A$), identifies the edge with the lowest confidence score in each cycle, and removes it until the graph is strictly acyclic ($\text{is\_dag} == \text{True}$).
- **Transitive Reduction**: If there is an alternate directed path from $u$ to $v$ of length $\ge 2$, removes the redundant direct bypass edge $(u, v)$ to leave the clean essential prerequisite backbone.

#### 6. Stage 5: Pedagogical Curriculum Planner (`CurriculumPlannerAgent`)
- Applies Kahn's topological sort with category priority.
- Enforces **DAG Topological Tier Monotonicity**: For every prerequisite edge $u \to v$, $\text{tier}(v) \ge \text{tier}(u)$.
- Organizes the curriculum into 4 structured tiers:
  - **Tier 1: Foundational Mathematical & Signal Preliminaries**
  - **Tier 2: Algorithmic & Communication Foundations**
  - **Tier 3: Domain Concepts & Theoretical Setting**
  - **Tier 4: Target Paper Contribution & Innovations**
- Synthesizes an explainable "Why Needed" rationale for every concept based on its outgoing dependencies.

#### 7. Stage 6: Grounded Evidence Agent (`EvidenceGroundingAgent`)
- Indexes paper paragraphs and maps each prerequisite edge to:
  - Verbatim text quote from the paper.
  - Section heading (sanitizing pipe characters `|`).
  - Mathematical formula references.
  - Foundational bibliography citations.
- Uses punctuation-free keyword matching so acronyms and bracketed names match sentences accurately.

#### 8. Pipeline Orchestrator & CLI Runner (`src/agents/orchestrator.py` & `run.py`)
- Coordinates Stages 1 through 6 on any given paper.
- Exports structured JSON (`output/<id>_prerequisite_curriculum.json`) and formatted Markdown report (`output/<id>_curriculum_report.md`).
- Provides simple CLI:
  - `python3 run.py --input <paper_id>`
  - `python3 run.py --list`
  - `python3 run.py --all`

#### 9. Search Catalog Utility (`query.py`)
- Provides fast search across the paper library:
  - `python3 query.py --category Mathematics` (or Algorithms, Theories, Methods)
  - `python3 query.py --concept "GLRT" --details`
- Outputs matching papers, concept roles, exact file paths (PDF, XML, JSON, MD), and grounded in-text quotes.

#### 10. Multi-Core 1,000-PDF Batch Trainer (`batch_train.py`)
- Designed for scale: processes 1,000 PDFs in a single run.
- Uses `ThreadPoolExecutor` for concurrent GROBID conversions.
- Uses `ProcessPoolExecutor` across all CPU cores for parallel agent execution.
- Includes automatic resume / fault tolerance (skips already processed papers).
- Aggregates thousands of concept pairs across all papers and trains the global prerequisite classifier (`models/checkpoints/prereq_classifier.json`).
- Compiles a consolidated `output/corpus_catalog.json`.

#### 11. Automated Test Suite (`run_tests.py` & `tests/`)
- Comprehensive test suite covering all data models, agents, cycle elimination, transitive reduction, copyright rejection, topological monotonicity, and CLI utilities.
- Guarantees 100% test pass rate with zero runtime warnings.

---

### DELIVERABLES EXPECTED
Build all python modules cleanly, format all code with docstrings and type hints, verify using `python3 run_tests.py`, and ensure the `.gitignore` cleanly excludes all raw PDFs and bulk generated JSON files.
```
