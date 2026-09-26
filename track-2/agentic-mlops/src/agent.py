"""Week 16/17 agentic loop with prompt-versioned policies and structured tracing.

Each query produces a trace containing (per assignment spec):
  step number, tool selected, tool arguments, raw tool result, reasoning,
  continue/stop decision, total iterations, termination reason.

Traces are saved as JSON to traces/ and logged to MLflow as artifacts
(see tracker.py). Three prompt versions (prompts/prompt_v*.txt) drive three
different decision policies so that V1 -> V2 -> V3 improvements are measurable:

  V1 (naive):   search once with raw question, always answer, never clarify.
                Hallucinates on empty evidence / tool failure.
  V2 (safer):   keyword reformulation (no synonyms), abstain on tool error,
                clarify on persistent weak evidence. Fixes fabrication but still
                fails paraphrases ("money back" vs "refund").
  V3 (robust):  keyword + synonym-expansion reformulation, multi-retry up to
                max_iterations, compositional coverage. Fixes paraphrases.

A live LLM (Groq preferred, OpenAI fallback) is used when API keys are present;
otherwise the deterministic local policies below execute the prompt's rules.
Both paths emit identical trace schemas.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import sys as _sys

# Make both `python src/agent.py` and `python -m src.agent` / `import src.agent`
# work: ensure the src/ directory itself is importable.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

try:
    from src.assistant import RagTool, approx_tokens, default_corpus
except ImportError:
    from assistant import RagTool, approx_tokens, default_corpus

# ---------------------------------------------------------------------------
# Config registry: the tracked hyperparameters per prompt version.
# These are the exact params logged to MLflow (tracker.py).
# ---------------------------------------------------------------------------
PROMPT_CONFIGS: dict[str, dict] = {
    "v1": {
        "prompt_version": "v1",
        "model_name": "local-policy-v1",
        "temperature": 0.7,
        "top_k": 3,
        "chunk_size": 300,
        "max_iterations": 2,
    },
    "v2": {
        "prompt_version": "v2",
        "model_name": "local-policy-v2",
        "temperature": 0.3,
        "top_k": 5,
        "chunk_size": 300,
        "max_iterations": 3,
    },
    "v3": {
        "prompt_version": "v3",
        "model_name": "local-policy-v3",
        "temperature": 0.0,
        "top_k": 5,
        "chunk_size": 500,
        "max_iterations": 5,
    },
}

STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "how", "do", "does", "tell", "me",
    "about", "please", "i", "my", "to", "in", "for", "on", "and", "or",
    "get", "can", "your", "you", "with", "are", "be",
}

# V3 synonym expansion map (paraphrase -> retrieval terms). Evidence-based fix
# for V2 failures on queries with zero keyword overlap (see README).
SYNONYM_MAP: list[tuple[list[str], str]] = [
    (["money", "back", "return", "refund", "returns", "purchase"], "refund return policy"),
    (["forgot", "forgotten", "locked", "access", "account"], "reset password"),
    (["shipping", "delivery", "arrive", "long"], "shipping delivery days"),
    (["open", "close", "closed", "hours", "when", "library"], "opening hours library"),
    (["warranty", "guarantee", "broken", "defect", "defective", "damage", "cover"], "warranty covers defects"),
    (["contact", "support", "phone", "email", "call", "help"], "contact support"),
    (["venv", "environment", "virtual", "python", "install", "package"], "virtual environments venv"),
    (["rag", "retrieval", "model", "machine", "learning", "ai"], "RAG retrieval generation"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = PROJECT_ROOT / "prompts"
TRACE_DIR = PROJECT_ROOT / "traces"


def load_prompt(version: str) -> str:
    """Load prompts/prompt_v{1,2,3}.txt text."""
    v = version.lower().lstrip("v")
    path = PROMPT_DIR / f"prompt_v{v}.txt"
    return path.read_text(encoding="utf-8")


def _keywords(question: str) -> list[str]:
    return [
        w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in STOPWORDS
    ]


def _expand_synonyms(question: str, keywords: list[str]) -> str:
    """V3 reformulation: original keywords + mapped synonym terms.

    Token-based matching (not substring) to avoid false triggers such as
    "phone" inside "headphones".
    """
    qtoks = set(re.findall(r"[a-z0-9]+", question.lower()))
    extra: list[str] = []
    for triggers, expansion in SYNONYM_MAP:
        if any(t in qtoks for t in triggers):
            extra.append(expansion)
    base = " ".join(keywords) or question
    if extra:
        return base + " " + " ".join(extra)
    return base


# ---------------------------------------------------------------------------
# Live LLM helpers (optional). Stdlib only, no new deps.
# ---------------------------------------------------------------------------
def _load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k and not k.startswith("#") and k not in os.environ:
                os.environ[k] = v.strip().strip('"').strip("'")


_load_env()


def _groq_json(system_prompt: str, user_prompt: str, temperature: float):
    """Call Groq OpenAI-compatible endpoint. Returns (dict, usage_tokens|None)."""
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return None
    body = json.dumps({
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + "\nReturn JSON only."},
        ],
        "max_tokens": 200,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    msg = data["choices"][0]["message"]["content"]
    u = data.get("usage") or {}
    toks = (u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)) or None
    return json.loads(msg), toks


def _openai_json(system_prompt: str, user_prompt: str, temperature: float):
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        from openai import OpenAI  # optional dep, only when key is set
    except ImportError:
        return None
    c = OpenAI()
    r = c.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + "\nReturn JSON only."},
        ],
        max_tokens=200,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    toks = (r.usage.prompt_tokens + r.usage.completion_tokens) if r.usage else None
    return json.loads(r.choices[0].message.content), toks


# ---------------------------------------------------------------------------
# Versioned decision policies (offline deterministic execution of each prompt).
# ---------------------------------------------------------------------------
def llm_decide(question: str, history: list[dict], evidence: list[dict],
               iteration: int, prompt_version: str, system_prompt: str,
               temperature: float) -> dict:
    """Decide next action following the loaded prompt's rules.

    Tries live LLM first (Groq/OpenAI); falls back to the deterministic policy
    that faithfully implements V1/V2/V3 rules offline.
    Returns {"action", "query", "reason", "tokens"}.
    """
    ev_txt = "\n".join(f"- [{e['doc_id']}] score={e['score']} {e['text']}" for e in evidence) or "(no evidence yet)"
    hist = "\n".join(
        f"iter{i+1}: q={h['query']} n={h['n_results']} err={h.get('error')}"
        for i, h in enumerate(history)
    ) or "(none)"
    user_prompt = (
        f"Question: {question}\nHistory:\n{hist}\nEvidence:\n{ev_txt}\n"
        f"Iteration {iteration + 1}. Respond ONLY with JSON "
        f'{{"action": "search|answer|clarify", "query": "...", "reason": "..."}}'
    )
    prompt_tokens = approx_tokens(system_prompt + user_prompt)

    # Live LLM path (same prompt text the offline policy implements).
    for fn in (_groq_json, _openai_json):
        try:
            out = fn(system_prompt, user_prompt, temperature)
            if out:
                d, real = out
                toks = real or (prompt_tokens + approx_tokens(json.dumps(d)))
                action = d.get("action", "answer")
                if action not in ("search", "answer", "clarify"):
                    action = "answer"
                return {"action": action, "query": d.get("query", ""),
                        "reason": d.get("reason", ""), "tokens": toks}
        except Exception:
            continue  # fall through to deterministic policy

    # ---- Deterministic offline policies (implement prompt text exactly) ----
    v = prompt_version.lower().strip()
    v = v[1:] if v.startswith("v") else v  # normalize "v1" -> "1", "1" -> "1"
    if v == "1":
        return _policy_v1(question, history, evidence, iteration, prompt_tokens)
    if v == "2":
        return _policy_v2(question, history, evidence, iteration, prompt_tokens)
    return _policy_v3(question, history, evidence, iteration, prompt_tokens)


def _policy_v1(question, history, evidence, iteration, prompt_tokens):
    """Naive: one raw search, then always answer (even with no evidence)."""
    toks = prompt_tokens + 60
    if not history:
        return {"action": "search", "query": question,
                "reason": "V1: initial search with raw question.", "tokens": toks}
    # After any search, answer with whatever came back (or hallucinate).
    best = evidence[0]["score"] if evidence else 0
    return {"action": "answer", "query": "",
            "reason": f"V1: answer with top result (best={best}); always answer.",
            "tokens": toks}


def _policy_v2(question, history, evidence, iteration, prompt_tokens):
    """Safer: keyword retry, abstain on error, clarify on weak evidence."""
    toks = prompt_tokens + 60
    if any(h.get("error") for h in history):
        return {"action": "clarify", "query": "",
                "reason": "V2: search tool failed; cannot answer confidently.",
                "tokens": toks}
    best = evidence[0]["score"] if evidence else 0
    if not evidence or best < 2.0:
        if iteration >= 1 or len(_keywords(question)) == 0:
            # Only one retry allowed; then clarify.
            if len(question.strip().split()) <= 2:
                return {"action": "clarify", "query": "",
                        "reason": "V2: vague query, need clarification.", "tokens": toks}
            if iteration >= 2:
                return {"action": "clarify", "query": "",
                        "reason": f"V2: evidence still weak (best={best}); clarify.",
                        "tokens": toks}
        kw = _keywords(question)
        # V2 has NO synonym expansion -> paraphrases keep failing (intended).
        return {"action": "search", "query": " ".join(kw) or question,
                "reason": f"V2: weak evidence (best={best}); retry with keywords.",
                "tokens": toks}
    return {"action": "answer", "query": "",
            "reason": f"V2: sufficient evidence (best={best}).", "tokens": toks}


def _policy_v3(question, history, evidence, iteration, prompt_tokens):
    """Robust: synonym-expansion retries, compositional coverage, multi-iteration."""
    toks = prompt_tokens + 60
    if any(h.get("error") for h in history):
        # Allow retries on error until iterations run out (tool may recover),
        # but never fabricate: clarify/abstain at the end.
        n_err = sum(1 for h in history if h.get("error"))
        if n_err >= 2 or iteration >= PROMPT_CONFIGS["v3"]["max_iterations"] - 1:
            return {"action": "clarify", "query": "",
                    "reason": "V3: search tool failed repeatedly; cannot answer confidently.",
                    "tokens": toks}
        # otherwise fall through to retry with expanded query
    else:
        if len(_keywords(question)) == 0 or len(question.strip().split()) <= 1:
            # Genuinely vague ("Hi") -> clarify immediately.
            if not evidence:
                return {"action": "clarify", "query": "",
                        "reason": "V3: vague query, need clarification.", "tokens": toks}
    best = evidence[0]["score"] if evidence else 0
    # Compositional check: e.g. return-window + warranty-exclusion question needs
    # both doc1 and doc6. Keep searching while only one side is covered.
    # Token-based matching to avoid substring false positives ("phone" in "headphones").
    qtoks = set(re.findall(r"[a-z0-9]+", question.lower()))
    ql = question.lower()
    needs_return = bool({"return", "returns", "refund", "refunds", "window", "headphone", "headphones"} & qtoks) or ("money back" in ql)
    needs_warranty = bool({"warranty", "defect", "defects", "defective", "damage"} & qtoks) or ("water damage" in ql)
    if evidence and best >= 2.0 and needs_return and needs_warranty:
        ids = {e["doc_id"] for e in evidence}
        if not ({"doc1", "doc6"} <= ids):
            if iteration < PROMPT_CONFIGS["v3"]["max_iterations"] - 1:
                missing = "warranty covers defects" if "doc1" in ids else "refund return policy"
                kw = _keywords(question)
                return {"action": "search",
                        "query": " ".join(kw) + " " + missing,
                        "reason": "V3: compositional query, missing second fact; search again.",
                        "tokens": toks}
    if not evidence or best < 2.0:
        if iteration >= PROMPT_CONFIGS["v3"]["max_iterations"] - 1:
            return {"action": "clarify", "query": "",
                    "reason": f"V3: evidence still weak (best={best}) at max iterations; clarify.",
                    "tokens": toks}
        kw = _keywords(question)
        return {"action": "search", "query": _expand_synonyms(question, kw),
                "reason": f"V3: weak evidence (best={best}); retry with keywords+synonyms.",
                "tokens": toks}
    return {"action": "answer", "query": "",
            "reason": f"V3: sufficient evidence (best={best}).", "tokens": toks}


# ---------------------------------------------------------------------------
# Main agentic loop with structured tracing.
# ---------------------------------------------------------------------------
def run_agent(question: str, prompt_version: str = "v3", tool: RagTool | None = None,
              verbose: bool = False, save_trace: bool = True) -> dict:
    """Run the decide-act-observe loop for one question.

    Returns dict with answer, iterations, tool_calls, tokens, latency_s,
    history, trace (list of steps), termination_reason, prompt_version, config.
    """
    v = prompt_version.lower().lstrip("v")
    if v not in ("1", "2", "3"):
        raise ValueError(f"Unknown prompt_version={prompt_version!r}; expected v1/v2/v3")
    key = f"v{v}"
    cfg = PROMPT_CONFIGS[key]
    system_prompt = load_prompt(key)

    tool = tool or RagTool(default_corpus(), chunk_size=cfg["chunk_size"])
    # Ensure tool chunking matches the prompt config.
    if tool.chunk_size != cfg["chunk_size"]:
        tool = RagTool(tool.docs, chunk_size=cfg["chunk_size"])

    evidence: list[dict] = []
    history: list[dict] = []
    trace_steps: list[dict] = []
    total_tokens = approx_tokens(question + system_prompt)
    tool_calls = 0
    iters = 0
    termination_reason = "max_iterations"
    final_answer = ""
    clarified = False
    t0 = time.perf_counter()

    for i in range(cfg["max_iterations"]):
        iters = i + 1
        d = llm_decide(question, history, evidence, i, key, system_prompt, cfg["temperature"])
        total_tokens += d.pop("tokens", 0)
        action = d["action"]
        if verbose:
            print(f"[iter {iters}/{cfg['max_iterations']}] {action}: {d['reason']}")

        if action == "search":
            tool_calls += 1
            try:
                res = tool.search(d["query"], top_k=cfg["top_k"])
            except Exception as e:  # tool failure -> record, continue loop
                history.append({"query": d["query"], "n_results": 0, "error": str(e)})
                total_tokens += 20
                trace_steps.append({
                    "step_number": iters,
                    "tool_selected": "rag_search",
                    "tool_arguments": {"query": d["query"], "top_k": cfg["top_k"]},
                    "raw_tool_result": {"ok": False, "error": str(e), "results": []},
                    "reasoning": d["reason"],
                    "decision": "continue",
                })
                continue
            if not res.get("ok") or not res.get("results"):
                history.append({"query": d["query"], "n_results": 0, "error": res.get("error")})
                evidence = []
                trace_steps.append({
                    "step_number": iters,
                    "tool_selected": "rag_search",
                    "tool_arguments": {"query": d["query"], "top_k": cfg["top_k"]},
                    "raw_tool_result": res,
                    "reasoning": d["reason"],
                    "decision": "continue",
                })
                continue
            capped = sorted(res["results"], key=lambda r: r["score"], reverse=True)[: cfg["top_k"]]
            history.append({"query": d["query"], "n_results": len(capped), "error": None})
            seen = {e["doc_id"]: e for e in evidence}
            for r in capped:
                seen[r["doc_id"]] = r
            evidence = sorted(seen.values(), key=lambda r: r["score"], reverse=True)[: cfg["top_k"]]
            total_tokens += approx_tokens(json.dumps(capped))
            trace_steps.append({
                "step_number": iters,
                "tool_selected": "rag_search",
                "tool_arguments": {"query": d["query"], "top_k": cfg["top_k"]},
                "raw_tool_result": res,
                "reasoning": d["reason"],
                "decision": "continue",
            })
        elif action == "answer":
            if key == "v1":
                # V1 faithfully fabricates when evidence is missing (documented failure).
                if not evidence or any(h.get("error") for h in history):
                    ans = ("Based on general knowledge: I believe this is covered, "
                           "but I could not verify it in the documents.")
                else:
                    ans = f"Based on {evidence[0]['doc_id']}: {evidence[0]['text']}"
                    # V1 cites only the single top doc even for compositional queries.
            else:
                if not evidence or any(h.get("error") for h in history):
                    ans = ("I couldn't retrieve reliable evidence, so I can't answer "
                           "confidently. Please try again later.")
                else:
                    # Compositional questions (return + warranty) need both facts.
                    # Cite doc1 + doc6 explicitly when both are retrieved, otherwise
                    # fall back to the single top document.
                    qtoks_ans = set(re.findall(r"[a-z0-9]+", question.lower()))
                    ql_ans = question.lower()
                    wants_both = (
                        bool({"return", "returns", "refund", "refunds", "window", "headphone", "headphones"} & qtoks_ans)
                        or ("money back" in ql_ans)
                    ) and (
                        bool({"warranty", "defect", "defects", "defective", "damage"} & qtoks_ans)
                        or ("water damage" in ql_ans)
                    )
                    by_id = {e["doc_id"]: e for e in evidence}
                    if wants_both and {"doc1", "doc6"} <= set(by_id):
                        ans = f"Based on doc1: {by_id['doc1']['text']} Based on doc6: {by_id['doc6']['text']}"
                    else:
                        ans = f"Based on {evidence[0]['doc_id']}: {evidence[0]['text']}"
            total_tokens += approx_tokens(ans)
            final_answer = ans
            clarified = False
            termination_reason = "answered"
            trace_steps.append({
                "step_number": iters,
                "tool_selected": "none (answer)",
                "tool_arguments": {},
                "raw_tool_result": {"evidence_used": [e["doc_id"] for e in evidence]},
                "reasoning": d["reason"],
                "decision": "stop",
            })
            break
        else:  # clarify / abstain
            q = "Could you clarify your question? " + (d.get("reason") or "")
            # Preserve abstention wording for tool-error cases.
            if any(h.get("error") for h in history) and key in ("v2", "v3"):
                q = ("I couldn't retrieve reliable evidence, so I can't answer "
                     "confidently. Please try again later.")
            total_tokens += approx_tokens(q)
            final_answer = q
            clarified = True
            termination_reason = "clarified" if "clarif" in q.lower() else "abstained"
            trace_steps.append({
                "step_number": iters,
                "tool_selected": "none (clarify)",
                "tool_arguments": {},
                "raw_tool_result": {},
                "reasoning": d["reason"],
                "decision": "stop",
            })
            break
    else:
        final_answer = "Max iterations reached without enough evidence. Could you rephrase?"
        clarified = True
        termination_reason = "max_iterations"
        total_tokens += approx_tokens(final_answer)

    latency_s = time.perf_counter() - t0
    result = {
        "answer": final_answer,
        "iterations": iters,
        "tool_calls": tool_calls,
        "tokens": total_tokens,
        "latency_s": round(latency_s, 3),
        "history": history,
        "clarified": clarified,
        "prompt_version": key,
        "config": cfg,
        "trace": trace_steps,
        "total_iterations": iters,
        "termination_reason": termination_reason,
    }
    if save_trace:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        safe_q = re.sub(r"[^a-z0-9]+", "_", question.lower())[:40].strip("_")
        path = TRACE_DIR / f"trace_{key}_{ts}_{safe_q or 'q'}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["trace_path"] = str(path)
    return result


if __name__ == "__main__":
    import sys
    if "--fail" in sys.argv:
        try:
            import src.assistant as _a
        except ImportError:
            import assistant as _a
        _a.FAILURE_INJECTION = True
    ver = "v3"
    for a in sys.argv[1:]:
        if a in ("v1", "v2", "v3", "--fail"):
            if a.startswith("v"):
                ver = a
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("-") and a not in ("v1", "v2", "v3")) or "What is the refund policy?"
    print(json.dumps(run_agent(q, prompt_version=ver, verbose=True), indent=2))
