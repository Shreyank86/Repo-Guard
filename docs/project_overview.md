# RepoGuard Project Overview

## 1. Problem Statement
Large Language Models (LLMs) have demonstrated substantial utility in software engineering. However, when deployed to answer queries over complex software repositories, they suffer from critical failure modes:
1. **Hallucination of Code Entities**: Fabricating functions, classes, arguments, and module paths that do not exist.
2. **Unsupported Architectural Claims**: Asserting dependency relationships or call graphs that contradict the codebase.
3. **Over-Computation & Token Waste**: Executing multiple unguided search and generation cycles even when the target information is unreachable.
4. **Failure to Abstain**: Generating plausible-sounding yet erroneous answers rather than explicitly declining when evidence is lacking.

## 2. The RepoGuard Solution
RepoGuard implements a four-tiered architecture designed to ensure verifiable repository grounding:

```
[User Query]
     │
     ▼
[Risk-Conditioned RL Agent] ──(Learns step-by-step policy: SEARCH, INSPECT, VERIFY, ANSWER, ABSTAIN)
     │
     ├──► [Repository Intelligence] ── (Static AST, Symbol Table, Call Graph, FAISS Passages)
     ├──► [Frozen Code LLM]         ── (Generates grounded candidate answers with claim tags)
     └──► [Verification Engine]     ── (Fact-checks code claims against ground-truth AST)
     │
     ▼
[Verified Response or Principled Abstention]
```

## 3. Core Principles
* **Evidence-First**: Every answer must be linked to verifiable code snippets and line ranges.
* **Structural Ground Truth**: Natural language is ambiguous, but ASTs are deterministic. Verification evaluates verifiable code entities rather than fuzzy semantics.
* **Learned Risk Sensitivity**: An RL agent learns when to answer and when to abstain based on user risk tolerance and verification feedback.
* **Hardware Pragmatism**: Designed to run and be trained on realistic hardware (e.g. 6GB VRAM) by freezing the LLM and training only the decision policy.

