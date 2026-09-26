# Evaluation report — prompt v3

Config: model=local-policy-v3, temp=0.0, top_k=5, chunk=500, max_iter=5

Pass rate: **12/12 = 100%** | correctness_avg=0.99 | preservation_avg=0.73 | avg_tokens=2334 | avg_latency=0.000s

Evidently: evidently 0.7.23 installed; wrote evidently_input_v3.csv. Deterministic LLM-as-a-judge used (Correctness + Information Preservation); Evidently input CSV kept for audit.

| ID | Question | Passed | Correctness | Preservation | Iters | Tokens | Termination |
|---|---|---|---|---|---|---|---|
| refund | What is the refund policy? | PASS | 1.0 | 0.762 | 2 | 2381 | answered |
| shipping | How long does standard shipping take? | PASS | 1.0 | 0.78 | 2 | 2483 | answered |
| password | How do I reset my password? | PASS | 1.0 | 0.821 | 2 | 2369 | answered |
| library | What are the library opening hours? | PASS | 1.0 | 0.667 | 2 | 2456 | answered |
| warranty | What does the warranty cover? | PASS | 1.0 | 0.882 | 2 | 2370 | answered |
| support | How do I contact support? | PASS | 1.0 | 0.848 | 2 | 2352 | answered |
| venv | How do I create a Python virtual environment? | PASS | 1.0 | 0.765 | 2 | 2404 | answered |
| rag | What is RAG in machine learning? | PASS | 1.0 | 0.593 | 2 | 2397 | answered |
| paraphrase-refund | How do I get my money back? | PASS | 1.0 | 0.762 | 2 | 2382 | answered |
| paraphrase-password | Forgotten credentials, locked out — need entry again | PASS | 1.0 | 0.821 | 2 | 2400 | answered |
| vague | Hi | PASS | 1.0 | 0.435 | 1 | 1473 | clarified |
| compositional | What is your return window for defective headphones with wat | PASS | 0.85 | 0.594 | 2 | 2535 | answered |
