# Evaluation report — prompt v2

Config: model=local-policy-v2, temp=0.3, top_k=5, chunk=300, max_iter=3

Pass rate: **9/12 = 75%** | correctness_avg=0.78 | preservation_avg=0.60 | avg_tokens=1620 | avg_latency=0.000s

Evidently: evidently 0.7.23 installed; wrote evidently_input_v2.csv. Deterministic LLM-as-a-judge used (Correctness + Information Preservation); Evidently input CSV kept for audit.

| ID | Question | Passed | Correctness | Preservation | Iters | Tokens | Termination |
|---|---|---|---|---|---|---|---|
| refund | What is the refund policy? | PASS | 1.0 | 0.762 | 2 | 1547 | answered |
| shipping | How long does standard shipping take? | PASS | 1.0 | 0.78 | 2 | 1562 | answered |
| password | How do I reset my password? | PASS | 1.0 | 0.821 | 2 | 1539 | answered |
| library | What are the library opening hours? | PASS | 1.0 | 0.667 | 2 | 1621 | answered |
| warranty | What does the warranty cover? | PASS | 1.0 | 0.882 | 2 | 1534 | answered |
| support | How do I contact support? | PASS | 1.0 | 0.848 | 2 | 1517 | answered |
| venv | How do I create a Python virtual environment? | PASS | 1.0 | 0.765 | 2 | 1568 | answered |
| rag | What is RAG in machine learning? | PASS | 1.0 | 0.593 | 2 | 1559 | answered |
| paraphrase-refund | How do I get my money back? | FAIL | 0.0 | 0.0 | 3 | 1971 | clarified |
| paraphrase-password | Forgotten credentials, locked out — need entry again | FAIL | 0.0 | 0.0 | 3 | 2024 | clarified |
| vague | Hi | PASS | 1.0 | 0.435 | 2 | 1427 | clarified |
| compositional | What is your return window for defective headphones with wat | FAIL | 0.333 | 0.6 | 2 | 1574 | answered |
