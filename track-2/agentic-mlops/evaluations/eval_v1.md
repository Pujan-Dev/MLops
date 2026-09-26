# Evaluation report — prompt v1

Config: model=local-policy-v1, temp=0.7, top_k=3, chunk=300, max_iter=2

Pass rate: **8/12 = 67%** | correctness_avg=0.69 | preservation_avg=0.59 | avg_tokens=948 | avg_latency=0.000s

Evidently: evidently 0.7.23 installed; wrote evidently_input_v1.csv. Deterministic LLM-as-a-judge used (Correctness + Information Preservation); Evidently input CSV kept for audit.

| ID | Question | Passed | Correctness | Preservation | Iters | Tokens | Termination |
|---|---|---|---|---|---|---|---|
| refund | What is the refund policy? | PASS | 1.0 | 0.762 | 2 | 1036 | answered |
| shipping | How long does standard shipping take? | PASS | 1.0 | 0.78 | 2 | 970 | answered |
| password | How do I reset my password? | PASS | 1.0 | 0.821 | 2 | 873 | answered |
| library | What are the library opening hours? | PASS | 1.0 | 0.667 | 2 | 1042 | answered |
| warranty | What does the warranty cover? | PASS | 1.0 | 0.882 | 2 | 1029 | answered |
| support | How do I contact support? | PASS | 1.0 | 0.848 | 2 | 849 | answered |
| venv | How do I create a Python virtual environment? | PASS | 1.0 | 0.765 | 2 | 1067 | answered |
| rag | What is RAG in machine learning? | PASS | 1.0 | 0.593 | 2 | 1062 | answered |
| paraphrase-refund | How do I get my money back? | FAIL | 0.0 | 0.056 | 2 | 789 | answered |
| paraphrase-password | Forgotten credentials, locked out — need entry again | FAIL | 0.0 | 0.118 | 2 | 814 | answered |
| vague | Hi | FAIL | 0.0 | 0.129 | 2 | 764 | answered |
| compositional | What is your return window for defective headphones with wat | FAIL | 0.333 | 0.6 | 2 | 1077 | answered |
