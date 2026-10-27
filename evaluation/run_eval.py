"""Evaluate the chatbot against evaluation/dataset.json.

Three stages, each scored separately so a drop can be traced to its cause:
  routing    - does the intent classifier pick the expected routes?
  retrieval  - is the expected policy section in the top 5 after hybrid search and reranking? (hit@5, MRR)
  answer     - does the full chatbot answer contain the expected facts, citing only the customer's own documents?

Needs Ollama with the models pulled, an indexed vector store and the customer API running (CUSTOMER_API_URL).
Exits with status 1 when a score is below its threshold.

    python -m evaluation.run_eval
    python -m evaluation.run_eval --stages routing,retrieval      # fast: no answer generation
    python -m evaluation.run_eval --ids sneha-claim-status,amit-lapsed
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv

load_dotenv()

from rag.chatbot import MAIN_MODEL, Chatbot  # noqa: E402  (needs .env loaded first)
from rag.intent import INTENT_MODEL, classify  # noqa: E402
from rag.retriever import RERANKER_MODEL, retrieve  # noqa: E402

HERE = Path(__file__).parent
DATASET = HERE / "dataset.json"
RESULTS = HERE / "results"
TOP_K = 5


def normalize(text: str) -> str:
    """Lower-case, drop thousands separators and markdown emphasis, collapse whitespace."""
    text = text.lower().replace("’", "'").replace(",", "").replace("*", "")
    return re.sub(r"\s+", " ", text)


def missing_facts(answer: str, must_include: list[list[str]]) -> list[list[str]]:
    """Each group lists acceptable phrasings of one fact; return the groups with no match."""
    answer = normalize(answer)
    return [group for group in must_include if not any(normalize(option) in answer for option in group)]


def routing_ok(expected: list[str], predicted: list[str]) -> bool:
    if expected == ["general"]:
        return predicted == ["general"]
    return set(expected) <= set(predicted)


def customer_documents(customer_id: str) -> list[str]:
    bot = Chatbot(customer_id)
    return bot._customer_sources(bot._get(f"/customers/{customer_id}/policies"))


def eval_case(case: dict, stages: set[str]) -> dict:
    result = {"id": case["id"]}
    sources = customer_documents(case["customer_id"])

    if "routing" in stages:
        predicted = classify(case["question"], [])["routes"]
        result["routing"] = {"expected": case["routes"], "predicted": predicted,
                             "ok": routing_ok(case["routes"], predicted)}

    if "retrieval" in stages and case["sections"]:
        found = [c["metadata"]["section"] for c in retrieve(case["question"], sources, top_k=TOP_K)]
        rank = next((i + 1 for i, s in enumerate(found) if s in case["sections"]), None)
        result["retrieval"] = {"expected": case["sections"], "found": found, "rank": rank,
                               "hit": rank is not None, "rr": 1 / rank if rank else 0.0}

    if "answer" in stages:
        bot = Chatbot(case["customer_id"])
        start = time.perf_counter()
        answer = "".join(bot.ask(case["question"]))
        seconds = time.perf_counter() - start
        allowed = {PurePosixPath(s).stem.replace("_", " ") for s in sources}
        foreign = sorted({s["title"] for s in bot.last_turn["sources"]} - allowed)
        missing = missing_facts(answer, case["must_include"])
        result["answer"] = {"text": answer, "seconds": round(seconds, 1), "missing": missing,
                            "foreign_sources": foreign, "scored": bool(case["must_include"]),
                            "ok": not missing and not foreign}
    return result


def summarize(results: list[dict]) -> dict:
    def rate(values):
        return round(sum(values) / len(values), 3) if values else None

    routing = [r["routing"]["ok"] for r in results if "routing" in r]
    retrieval = [r["retrieval"] for r in results if "retrieval" in r]
    answers = [r["answer"] for r in results if "answer" in r]
    return {
        "routing_accuracy": rate(routing),
        "retrieval_hit_at_5": rate([r["hit"] for r in retrieval]),
        "retrieval_mrr": rate([r["rr"] for r in retrieval]),
        "answer_accuracy": rate([a["ok"] for a in answers if a["scored"]]),
        "isolation_violations": sum(bool(a["foreign_sources"]) for a in answers),
        "answer_seconds_avg": rate([a["seconds"] for a in answers]),
        "cases": len(results),
    }


def mark(ok):
    return "PASS" if ok else "FAIL"


def report(summary: dict, results: list[dict], thresholds: dict, failures: list[str]) -> str:
    lines = [
        "# Chatbot evaluation",
        "",
        f"Models: intent `{INTENT_MODEL}`, answer `{MAIN_MODEL}`, reranker `{RERANKER_MODEL}`",
        "",
        "| Metric | Score | Threshold |",
        "|---|---|---|",
    ]
    for metric, value in summary.items():
        if value is not None:
            limit = thresholds.get(metric)
            lines.append(f"| {metric} | {value} | {'' if limit is None else limit} |")
    lines += ["", f"**Result: {'FAIL - ' + '; '.join(failures) if failures else 'PASS'}**", "",
              "| Case | Routing | Retrieval (rank) | Answer | Notes |", "|---|---|---|---|---|"]
    for r in results:
        notes = []
        routing = r.get("routing")
        if routing and not routing["ok"]:
            notes.append(f"routes {routing['predicted']}")
        retrieval = r.get("retrieval")
        if retrieval and not retrieval["hit"]:
            notes.append(f"top-5 {retrieval['found']}")
        answer = r.get("answer")
        if answer and answer["missing"]:
            notes.append(f"missing {answer['missing']}")
        if answer and answer["foreign_sources"]:
            notes.append(f"other customer's document {answer['foreign_sources']}")
        lines.append(" | ".join([
            f"| {r['id']}",
            mark(routing["ok"]) if routing else "-",
            (f"{mark(retrieval['hit'])} ({retrieval['rank'] or '-'})") if retrieval else "-",
            (mark(answer["ok"]) if answer["scored"] or answer["foreign_sources"] else "n/a") if answer else "-",
            "; ".join(notes).replace("|", "/") + " |",
        ]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", default="routing,retrieval,answer")
    parser.add_argument("--ids", help="comma-separated case ids to run (default: all)")
    parser.add_argument("--min-routing", type=float, default=0.85)
    parser.add_argument("--min-hit", type=float, default=0.9)
    parser.add_argument("--min-mrr", type=float, default=0.75)
    parser.add_argument("--min-answer", type=float, default=0.75)
    args = parser.parse_args()

    stages = set(args.stages.split(","))
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}", flush=True)
        results.append(eval_case(case, stages))

    summary = summarize(results)
    thresholds = {"routing_accuracy": args.min_routing, "retrieval_hit_at_5": args.min_hit,
                  "retrieval_mrr": args.min_mrr, "answer_accuracy": args.min_answer, "isolation_violations": 0}
    failures = [f"{metric} {summary[metric]} < {limit}" for metric, limit in thresholds.items()
                if metric != "isolation_violations" and summary[metric] is not None and summary[metric] < limit]
    if summary["isolation_violations"]:
        failures.append(f"{summary['isolation_violations']} answer(s) cited another customer's document")

    text = report(summary, results, thresholds, failures)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "report.md").write_text(text, encoding="utf-8")
    (RESULTS / "results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2),
                                          encoding="utf-8")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text)

    print("\n" + text)
    print(f"Saved {RESULTS / 'report.md'} and results.json")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
