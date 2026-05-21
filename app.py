"""
ContextGuard — Flask Web UI
Group 2: Diyar Buyuksahin, Etem Tolga Erten, Süleyman Kılıç, Andrew Mabuto

Endpoints:
  GET  /                  → Ana sayfa (UI)
  POST /api/run           → Pipeline çalıştır (JSON)
  POST /api/benchmark     → 4-koşul benchmark (JSON, SSE stream)
  GET  /api/health        → Health check
"""

import json
import os
import time
import traceback

from flask import Flask, render_template, request, jsonify, Response, stream_with_context

from contextguard.mcp_governance import MCPGovernance
from contextguard.context_store import ContextStore
from contextguard.agents import (
    ContextRetrievalAgent,
    ReasoningAgent,
    GroundingValidatorAgent,
    AdversarialTesterAgent,
)
from contextguard.metrics import evaluate_response

app = Flask(__name__)


# ── Helper ────────────────────────────────────────────────────────────────────

def parse_docs(raw: str) -> list[str]:
    """Split textarea input into a list of documents (newline-separated)."""
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    return lines


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "llama-3.3-70b-versatile"})


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """
    Run the full ContextGuard pipeline on a query + document pool.
    Returns a JSON result suitable for the UI.
    """
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    docs_raw = data.get("documents") or ""
    docs  = parse_docs(docs_raw) if isinstance(docs_raw, str) else docs_raw
    num_adv = int(data.get("adversarial_trials", 2))

    if not query:
        return jsonify({"error": "Query is required."}), 400
    if not docs:
        return jsonify({"error": "At least one document is required."}), 400

    try:
        mcp   = MCPGovernance()
        store = ContextStore()

        # Agent 1
        retrieval_agent = ContextRetrievalAgent(mcp, store)
        retrieved, vid  = retrieval_agent.retrieve_and_rank(query, docs)
        if not retrieved:
            return jsonify({"error": "No relevant context found for this query."}), 400

        # Agent 2
        reasoning_agent = ReasoningAgent(mcp, store)
        answer          = reasoning_agent.generate_answer(query, retrieved, version_id=vid)

        # Agent 3
        validator_agent   = GroundingValidatorAgent(mcp, store)
        validation_result = validator_agent.validate(query, answer, retrieved)

        # Agent 4
        adversarial_agent = AdversarialTesterAgent(mcp, store)
        robustness_result = adversarial_agent.test_robustness(
            query, retrieved, reasoning_agent, num_trials=num_adv
        )

        # Metrics
        metrics = evaluate_response(
            answer=answer,
            context=retrieved,
            validation=validation_result,
            robustness=robustness_result,
        )

        return jsonify({
            "query":             query,
            "answer":            answer,
            "retrieved_context": retrieved,
            "version_id":        vid,
            "validation":        validation_result,
            "robustness":        robustness_result,
            "metrics":           metrics,
            "mcp_log":           mcp.get_execution_log(),
            "store_history":     store.history(),
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    """
    Run the 4-condition benchmark and stream Server-Sent Events back to the UI.
    Each SSE message is one JSON line (query result or summary).
    """
    from evaluate import (
        EVAL_DATASET, baseline_rag, single_agent_llm,
        multiagent_no_mcp, full_contextguard, keyword_accuracy
    )

    def generate():
        results = []
        for idx, item in enumerate(EVAL_DATASET, 1):
            query    = item["query"]
            docs     = item["docs"]
            keywords = item["expected_keywords"]

            try:
                ans_a = baseline_rag(query, docs)
                acc_a = keyword_accuracy(ans_a, keywords)

                ans_b = single_agent_llm(query)
                acc_b = keyword_accuracy(ans_b, keywords)

                ans_c = multiagent_no_mcp(query, docs)
                acc_c = keyword_accuracy(ans_c, keywords)

                res_d  = full_contextguard(query, docs)
                ans_d  = res_d.get("answer", "")
                acc_d  = keyword_accuracy(ans_d, keywords)
                met_d  = res_d.get("metrics", {})

                row = {
                    "idx":   idx,
                    "query": query,
                    "A":     {"accuracy": acc_a, "answer": ans_a},
                    "B":     {"accuracy": acc_b, "answer": ans_b},
                    "C":     {"accuracy": acc_c, "answer": ans_c},
                    "D":     {
                        "accuracy": acc_d,
                        "answer":   ans_d,
                        **met_d,
                    },
                }
                results.append(row)
                yield f"data: {json.dumps({'type': 'row', 'data': row})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'query': query, 'msg': str(e)})}\n\n"

        # Summary
        if results:
            summary = {}
            for cond in ["A", "B", "C", "D"]:
                avg_acc = sum(r[cond]["accuracy"] for r in results) / len(results)
                summary[cond] = {"avg_accuracy": round(avg_acc, 4)}
            for key in ["hallucination_rate", "cis_score", "cue_score", "robustness_score", "overall_score"]:
                vals = [r["D"].get(key, 0) for r in results]
                summary["D"][key] = round(sum(vals) / len(vals), 4)

            yield f"data: {json.dumps({'type': 'summary', 'data': summary})}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  ContextGuard Web UI → http://localhost:{port}\n")
    app.run(debug=True, port=port, threaded=True)
