# Scientific Prerequisite Concept Extraction & Knowledge Graph System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Test Suite](https://img.shields.io/badge/tests-100%25%20passing-brightgreen.svg)]()
[![Zero Dependencies](https://img.shields.io/badge/dependencies-pure%20std%20library-green.svg)]()

A production-grade, modular Multi-Agent System designed to automatically determine the foundational prerequisite concepts (Mathematics, Algorithms, Theories, Methods) required before reading an academic research paper, construct a Prerequisite Knowledge Graph (Directed Acyclic Graph), generate an optimal topological learning sequence (pedagogical curriculum), and ground every prerequisite in verifiable textual evidence and mathematical formulas.

---

## Table of Contents
- [Core Problem Statement](#core-problem-statement)
- [Theoretical Foundations](#theoretical-foundations)
- [Critical Anti-Patterns Avoided](#critical-anti-patterns-avoided)
- [System Architecture](#system-architecture)
- [Directory Layout](#directory-layout)
- [Quickstart & Usage](#quickstart--usage)
  - [1. Single Paper Execution](#1-single-paper-execution)
  - [2. Library Listing & Batch Runner](#2-library-listing--batch-runner)
  - [3. Deep Catalog Search (`query.py`)](#3-deep-catalog-search-querypy)
  - [4. Multi-Core 1,000-PDF Batch Trainer (`batch_train.py`)](#4-multi-core-1000-pdf-batch-trainer-batch_trainpy)
  - [5. Master Test Suite (`run_tests.py`)](#5-master-test-suite-run_testspy)
- [Mathematical & Algorithmic Specifications](#mathematical--algorithmic-specifications)
- [Artifacts & Outputs](#artifacts--outputs)

---

## Core Problem Statement
Given an academic research paper (PDF or GROBID TEI-XML), automatically determine the foundational prerequisite concepts required before reading the paper, construct a Prerequisite Knowledge Graph (DAG), generate an optimal topological learning sequence (pedagogical curriculum), and ground every prerequisite in verifiable textual evidence and mathematical formulas from the paper.

---

## Theoretical Foundations
This system directly operationalizes algorithms and theoretical constraints from two seminal research works:

1. **"Exploring knowledge graphs for the identification of concept prerequisites"** (Manrique, Nunes, & Mariño, 2019):
   - **Reflected Reference Distance (RefD)** for asymmetric directional prerequisite learning:
     $$\text{RefD}(A, B) = \frac{R(B \to A) - R(A \to B)}{\max(R(B \to A), R(A \to B))}$$
     Measures reference asymmetry: dependent concepts refer back to foundational concepts, but foundational concepts do not reference downstream applications.
   - Graph topological properties (centrality, reachability, DAG acyclicity).

2. **"Concept Extraction and Prerequisite Relation Learning from Educational Data"** (Pan et al., AAAI 2019 / iPRL framework):
   - Hybrid prerequisite scoring combining feature classification:
     - **Taxonomic Hierarchy Delta:** $\text{tier}(\text{Math}) < \text{tier}(\text{Theories}) < \text{tier}(\text{Algorithms}) < \text{tier}(\text{Methods})$
     - **Sequential Position Offset:** Concepts introduced in early sections (Preliminaries/Intro) precede Methodology/Contribution.
     - **Lexical Subsumption:** If concept $A$'s name is a component of $B$'s name (e.g. *Oblique Projection* in *Oblique Projection-Based Jamming Cancellation*), $A \to B$.
   - Global graph recovery constraints ensuring acyclicity, transitivity reduction, and pedagogical monotonicity.

---

## Critical Anti-Patterns Avoided
1. **DO NOT use generic spaCy `en_core_web_sm`:** Generic spaCy models extract people, single letters ('H', 'K'), dates, and table numbers instead of scientific concepts. We employ a dedicated 4-way scientific taxonomy with dynamic scientific phrase mining.
2. **DO NOT perform pairwise $O(N^2)$ character slicing between entities:** Slicing creates hundreds of megabytes of noisy strings ("is", "are", "consists of") that lack prerequisite directionality. We use structured feature extraction, RefD asymmetry, and epistemic dependency logic.
3. **DO NOT ignore `<tei:formula>` mathematical equation tags:** Mathematical equations contain crucial foundational operators (e.g. projection operators $E_{sr}$, Fourier matrices $F$, covariance matrices $R_{xx}$, likelihood ratios $\Lambda(y)$). The system parses and indexes every formula.
4. **DO NOT accept copyright disclaimers as paper titles:** GROBID frequently misidentifies ACM/IEEE copyright text ("Permission to make digital...") as paper titles. We implement a regex rejection filter and heading fallback.
5. **DO NOT violate DAG topological tier monotonicity:** In the curriculum, if concept $A$ is a prerequisite for concept $B$ ($A \to B$), then $\text{tier}(B) \ge \text{tier}(A)$. A student is never told to learn Tier 3 before its prerequisite in Tier 2.
6. **DO NOT use fragile external heavy dependencies:** All core graph algorithms (DFS cycle detection, Greedy Feedback Arc Set, transitive reduction, Kahn's topological sort) are built using pure Python standard library (`collections`, `dataclasses`, `xml.etree.ElementTree`, `re`, `pathlib`, `argparse`, `concurrent.futures`) guaranteeing zero installation fragility.

---

## System Architecture

```text
 Academic Research Paper (PDF or GROBID TEI-XML)
                     │
                     ▼
  [Stage 1: PaperParserAgent]
  • XML Parsing & Copyright Rejection
  • Formula Extraction (<tei:formula>) & Citation Mapping
                     │
                     ▼
  [Stage 2: ConceptExtractorAgent]
  • 4-Way Taxonomy (Mathematics, Algorithms, Theories, Methods)
  • Formula Operator Mining & Dynamic Scientific Terms / Acronyms
  • Paper Contribution Disambiguation
                     │
                     ▼
  [Stage 3: PrerequisiteRelationAgent]
  • Asymmetric RefD & iPRL Scoring
  • Taxonomic Delta, Sequential Offset, Lexical Subsumption
  • Epistemic Rationale Synthesis
                     │
                     ▼
  [Stage 4: GraphReasoningAgent]
  • Directed Graph Construction
  • Greedy Feedback Arc Set (FAS) DFS Cycle Elimination
  • Transitive Reduction (Essential Backbone Isolation)
                     │
                     ▼
  [Stage 5: CurriculumPlannerAgent]
  • Kahn's Topological Sort with Category Priority
  • Strict DAG Monotonicity Enforcement (tier(v) >= tier(u))
  • 4-Tier Curriculum Organization (Why Needed Generation)
                     │
                     ▼
  [Stage 6: EvidenceGroundingAgent]
  • Punctuation-Free In-Text Quote Matching
  • Formula Reference Linking & Citation Grounding
  • Section Heading Pipe Sanitization ('|' -> '-')
                     │
                     ▼
  [Outputs Generated]
  • output/<id>_prerequisite_curriculum.json
  • output/<id>_curriculum_report.md
  • output/<id>_graph.mmd (Mermaid Flowchart)
```

---

## Directory Layout
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

## Quickstart & Usage

The system runs out-of-the-box using standard Python 3.8+:

### 1. Single Paper Execution
Run the full 6-stage multi-agent pipeline on a paper ID, XML file path, or raw PDF:
```bash
# Run on an existing TEI-XML paper:
python3 run.py --input paper_opjc

# Run directly on a raw PDF (automatically converts to GROBID TEI-XML first):
python3 run.py --input data/raw_pdfs/sample_paper_viterbi.pdf
```

Outputs generated:
- `output/<id>_prerequisite_curriculum.json`
- `output/<id>_curriculum_report.md`
- `output/<id>_graph.mmd`

### 2. Raw PDF Batch Conversion
Batch convert all raw PDFs in `data/raw_pdfs/` to GROBID TEI-XML format:
```bash
python3 run.py --convert-pdfs
```

### 2. Library Listing & Batch Runner
List all available papers in the library and their processed status:
```bash
python3 run.py --list
```

Process all papers in `data/grobid_output/` sequentially:
```bash
python3 run.py --all
```

### 3. Deep Catalog Search & Ontology Identification (`query.py`)
Search papers by category:
```bash
python3 query.py --category Mathematics
```

Search papers by concept name or acronym with deep grounded quotes, formulas, and exact file paths:
```bash
python3 query.py --concept "GLRT" --details
```

Classify and identify any concept directly against the 14,000+ topics in the **Computer Science Ontology** (`CSO.3.3.owl`):
```bash
python3 query.py --cso "GLRT"
python3 query.py --cso "Fourier Transform"
```

### 4. Multi-Core 1,000-PDF Batch Trainer (`batch_train.py`)
Processes hundreds to 1,000 papers across all CPU cores with automatic resume, aggregates concept pairs, trains the global classifier checkpoint (`models/checkpoints/prereq_classifier.json`), and compiles `output/corpus_catalog.json`:
```bash
python3 batch_train.py
```

### 5. Master Test Suite (`run_tests.py`)
Execute the complete test suite covering schemas, all 6 agents, graph reasoning, monotonicity, and CLI tools:
```bash
python3 run_tests.py
```

### 6. Interactive Step-by-Step Feature Demonstration (`demo.py`)
Walk through every algorithm and agent stage one by one with live outputs:
```bash
python3 demo.py --all           # Run all 7 feature demonstrations
python3 demo.py --step 4        # Run only Step 4 (Greedy FAS cycle removal & Transitive Reduction)
python3 demo.py --interactive   # Guided tour with pauses between steps
```

---

## Mathematical & Algorithmic Specifications

### 1. Greedy Feedback Arc Set (FAS)
To guarantee strict DAG acyclicity ($\text{is\_dag} == \text{True}$):
1. Locate directed cycles ($A \to B \to \dots \to A$) using 3-color DFS traversal.
2. In each cycle, identify the edge with the minimum confidence and asymmetry score:
   $$e^* = \arg\min_{e \in \text{cycle}} (\text{confidence}(e), \text{asymmetry}(e))$$
3. Remove $e^*$ and repeat until the graph has zero cycles.

### 2. Transitive Reduction
Removes redundant bypass edges $(u, v)$ when an alternate directed path of length $\ge 2$ exists ($u \to \dots \to v$), yielding the clean essential prerequisite backbone.

### 3. DAG Topological Tier Monotonicity
For every prerequisite edge $u \to v$:
$$\text{tier}(v) \ge \text{tier}(u)$$
Enforced by computing Kahn's topological sort with category priority (Math > Algorithms > Theories > Methods), assigning tiers in topological order:
$$\text{tier}(v) = \max\left(\text{natural\_tier}(v), \max_{u \in \text{parents}(v)} \text{tier}(u)\right)$$
and anchoring target paper contributions strictly to Tier 4.

---

## License
MIT License. Built for production-grade academic research and knowledge graph construction.
