"""Week 15 baseline assistant: local TF-IDF RAG/search tool with chunking.

Reused as the retrieval tool for the Week 16/17 agentic loop. Stdlib only so the
project runs offline; live LLMs (Groq/OpenAI) are optional in agent.py.

Chunking: documents longer than `chunk_size` are split into overlapping chunks so
that prompt configs can experiment with chunk_size (300 vs 500). Each chunk keeps
its parent doc_id as `<doc_id>#c<i>` but the agent cites the parent doc_id.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Global failure-injection switch. When True, search() raises TimeoutError to
# simulate a retrieval outage. Toggled by `--fail` flags / tests.
FAILURE_INJECTION = False

_STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "how", "do", "does", "tell", "me",
    "about", "please", "i", "my", "to", "in", "for", "on", "and", "or",
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def approx_tokens(text: str) -> int:
    """Closest reliable offline token estimate (~1 token / 4 chars)."""
    return max(1, len(text) // 4)


def chunk_text(doc_id: str, text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split a document into overlapping char chunks.

    Args:
        doc_id: parent document id.
        text: full document text.
        chunk_size: max chars per chunk.
        overlap: chars of overlap between consecutive chunks.

    Returns:
        List of {"chunk_id", "doc_id", "text"} dicts.
    """
    if len(text) <= chunk_size:
        return [{"chunk_id": f"{doc_id}#c0", "doc_id": doc_id, "text": text}]
    chunks: list[dict] = []
    start = 0
    idx = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        piece = text[start : start + chunk_size]
        chunks.append({"chunk_id": f"{doc_id}#c{idx}", "doc_id": doc_id, "text": piece})
        if start + chunk_size >= len(text):
            break
        start += step
        idx += 1
    return chunks


def default_corpus() -> dict[str, str]:
    """Week 15 baseline corpus: 8 support docs (doc1..doc8)."""
    return {
        "doc1": "The refund policy allows returns within 30 days of purchase with a receipt. Refunds are issued to the original payment method.",
        "doc2": "Shipping takes 3-5 business days for standard delivery. Express shipping delivers in 1-2 business days. Shipping is free over $50.",
        "doc3": "To reset your password, go to Settings > Account > Reset password. A reset link is emailed and expires in 24 hours.",
        "doc4": "The library opening hours are 9am to 8pm Monday to Friday, and 10am to 4pm on Saturday. Closed on Sunday.",
        "doc5": "Machine learning models include regression, classification, and clustering. RAG combines retrieval with generation for grounded answers.",
        "doc6": "Warranty covers manufacturing defects for 12 months. It does not cover accidental damage or water damage.",
        "doc7": "Contact support at support@example.com or call 555-0100 between 9am and 5pm on weekdays.",
        "doc8": "Python virtual environments isolate dependencies. Create one with python -m venv env and activate it before installing packages.",
    }


class RagTool:
    """TF-IDF retriever over chunked documents.

    Attributes:
        docs: {doc_id: full text} mapping.
        chunk_size: chars per retrieval chunk.
    """

    def __init__(self, docs: dict[str, str] | None = None, chunk_size: int = 500):
        self.docs: dict[str, str] = docs or default_corpus()
        self.chunk_size = chunk_size
        # Build chunk index.
        self.chunks: list[dict] = []
        for did, text in self.docs.items():
            self.chunks.extend(chunk_text(did, text, chunk_size=chunk_size))
        # Precompute IDF over chunks.
        n = len(self.chunks)
        df: Counter[str] = Counter()
        self._tok: list[list[str]] = []
        for ch in self.chunks:
            toks = tokenize(ch["text"])
            self._tok.append(toks)
            for w in set(toks):
                df[w] += 1
        self.idf: dict[str, float] = {
            w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()
        }

    def _score(self, q_toks: list[str], d_toks: list[str]) -> float:
        tf = Counter(d_toks)
        return sum(tf[w] * self.idf.get(w, 0.0) for w in q_toks)

    def search(self, query: str, top_k: int = 5) -> dict:
        """Search chunks, then collapse to best chunk per parent doc.

        Returns {"ok", "results": [{"doc_id","chunk_id","score","text"}], "error"?}.
        Results are sorted by score desc, deduplicated by doc_id, capped at top_k.
        """
        if FAILURE_INJECTION:
            raise TimeoutError("Simulated search timeout (FAILURE_INJECTION=True)")
        if not query or not query.strip():
            return {"ok": False, "error": "empty query", "results": []}
        q = tokenize(query)
        scored = sorted(
            ((self._score(q, dt), i) for i, dt in enumerate(self._tok)),
            reverse=True,
        )
        # Keep best chunk per doc_id.
        best: dict[str, dict] = {}
        for score, i in scored:
            if score <= 0:
                continue
            ch = self.chunks[i]
            did = ch["doc_id"]
            if did not in best or score > best[did]["score"]:
                best[did] = {
                    "doc_id": did,
                    "chunk_id": ch["chunk_id"],
                    "score": round(score, 3),
                    "text": ch["text"][: self.chunk_size],
                }
        ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)[:top_k]
        return {"ok": True, "results": ranked}
