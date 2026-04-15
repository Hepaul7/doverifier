# DoVerifier: Symbolic Verification for LLM Causal Reasoning

DoVerifier is a **symbolic verification framework** for evaluating whether an LLM-generated causal expression is **formally valid / equivalent** under a given causal DAG, using **do-calculus** and **probability rules**.

Instead of relying on string match, BLEU, or LLM-as-a-judge, DoVerifier checks **derivability**: whether a predicted expression can be transformed into the target expression via a sequence of sound rewrite rules (do-calculus Rules 1–3 + probability transformations).

> 📄 **Paper (ACL Anthology):** https://aclanthology.org/2026.eacl-long.56/  
> 🧾 **Citation:** See below  
> 👩‍🔬 **Authors:** Paul He, Yinya Huang, Mrinmaya Sachan, Zhijing Jin


---
Current causal QA benchmarks often score models by surface similarity. But in causal inference, correctness depends on **semantic equivalence under a graph**.

DoVerifier:
-  Recovers *semantically correct* answers missed by exact match
-  Provides **sound** verification via formal rules
-  Complete under the rules of do calculus

If you find a bug or unexpected behavior, please let us know by opening an issue!
## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{he-etal-2026-uncovering,
    title = "Uncovering Hidden Correctness in {LLM} Causal Reasoning via Symbolic Verification",
    author = "He, Paul  and
      Huang, Yinya  and
      Sachan, Mrinmaya  and
      Jin, Zhijing",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-long.56/",
    doi = "10.18653/v1/2026.eacl-long.56",
    pages = "1231--1250",
    ISBN = "979-8-89176-380-7",
    abstract = "Large language models (LLMs) are increasingly applied to tasks involving causal reasoning. However, current benchmarks often rely on string matching or surface-level metrics that fail to assess whether a model{'}s output is formally valid under causal semantics. We propose DoVerifier, a symbolic verification framework that checks whether LLM-generated causal expressions are derivable from a given causal graph using rules from do-calculus and probability theory. This allows us to recover correct answers that would otherwise be marked incorrect due to superficial differences. Evaluations on synthetic data and causal QA benchmarks show that DoVerifier more accurately captures semantic correctness than standard metrics, offering a more rigorous and informative way to evaluate LLMs on causal tasks."
}
