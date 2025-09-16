# GPT-2 Text Poisoning Experiment

A fun experiment exploring text **“poisoning”** for GPT-2 embeddings.  
This project applies small semantic changes, token-boundary manipulations, invisible characters, and homoglyph substitutions to see how robust GPT-2 embeddings really are. All while keeping human readability.

---

## Results
- Lowest similarity achieved with GPT-2: **~0.8885**
- According to Claude’s analysis:
  - **Token-level similarity:** ~0.4–0.6  
  - **Character n-gram similarity:** ~0.3–0.5  
  - **Semantic embedding similarity:** ~0.6–0.8 (since meaning, imagery, and coherence are preserved)

> 🔎 Similarity can go **lower** if different tweaks or more aggressive poisoning strategies are applied.

---

## Notes

- Experiments show that small perturbations can drastically affect embeddings while still being human-readable.
- Certain approaches (e.g., homoglyphs + boundary attacks) yield more disruption than simple substitutions.
- Longer text sometimes exceeds GPT-2’s **1024 token limit**, so truncation or shorter inputs may be required.
