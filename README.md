# DoVerifier: Symbolic Verification for LLM Causal Reasoning

DoVerifier is a **symbolic verification framework** for evaluating whether an LLM-generated causal expression is **formally valid / equivalent** under a given causal DAG, using **do-calculus** and **probability rules**.

Instead of relying on string match, BLEU, or LLM-as-a-judge, DoVerifier checks **derivability**: whether a predicted expression can be transformed into the target expression via a sequence of sound rewrite rules (do-calculus Rules 1–3 + probability transformations).

> Paper: *Uncovering Hidden Correctness in LLM Causal Reasoning via Symbolic Verification* (EACL)  
> Authors: Paul He, Yinya Huang, Mrinmaya Sachan, Zhijing Jin

---
Current causal QA benchmarks often score models by surface similarity. But in causal inference, correctness depends on **semantic equivalence under a graph**.

DoVerifier:
-  Recovers *semantically correct* answers missed by exact match
-  Provides **sound** verification via formal rules
-  Complete under the rules of do calculus

If you find a bug or unexpected behavior, please let us know by opening an issue!
## Citation

If you use this code, please cite our paper (to appear at EACL):

```bibtex
@inproceedings{he2026uncovering,
  title     = {Uncovering Hidden Correctness in {LLM} Causal Reasoning via Symbolic Verification},
  author    = {Paul He and Yinya Huang and Mrinmaya Sachan and Zhijing Jin},
  booktitle = {Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics},
  year      = {2026}
}
