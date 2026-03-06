"""
Test Runner — 3 Evaluation Scenarios
Runs all 3 assignment queries, logs full traces to ./logs/
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

TEST_QUERIES = [
    {
        "id"   : "test_1_verification",
        "label": "Test 1 — Verification Challenge",
        "query": (
            "What is the total number of jobs reported in the "
            "Cyber Ireland 2022 report? Provide the exact page number "
            "and a verbatim citation from the document."
        ),
        "expected": "Exact integer + page number + verbatim quote. No hallucinations.",
    },
    {
        "id"   : "test_2_synthesis",
        "label": "Test 2 — Data Synthesis Challenge",
        "query": (
            "Compare the concentration of 'Pure-Play' cybersecurity firms "
            "in the South-West region against the National Average. "
            "Include exact figures and calculate the difference or ratio."
        ),
        "expected": "Navigate regional tables, extract Pure-Play metrics, mathematically accurate comparison.",
    },
    {
        "id"   : "test_3_forecasting",
        "label": "Test 3 — Forecasting Challenge",
        "query": (
            "Based on the 2022 baseline job figures and the stated 2030 "
            "job target in the report, what is the required Compound Annual "
            "Growth Rate (CAGR) to achieve that goal? Show your calculation."
        ),
        "expected": "Extract 2022 baseline + 2030 target, calculate CAGR = (end/start)^(1/years)-1 via calculator tool.",
    },
]


def run_all_tests():
    from graph import run_query

    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  Cyber Ireland 2022 — Multi-Agent Test Suite ║")
    logger.info("╚══════════════════════════════════════════════╝\n")

    ts      = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results = []

    for i, test in enumerate(TEST_QUERIES, 1):
        logger.info(f"\n{'═'*65}")
        logger.info(f"  {test['label']}  ({i}/3)")
        logger.info(f"{'═'*65}")
        logger.info(f"  Query: {test['query']}\n")

        result = run_query(test["query"])
        result.update({
            "test_id"      : test["id"],
            "test_label"   : test["label"],
            "expected"     : test["expected"],
        })

        # Save individual trace
        trace_path = LOG_DIR / f"{ts}_{test['id']}.json"
        with open(trace_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        icon = "✅" if result["status"] == "success" else "❌"
        logger.info(f"\n{icon}  {test['label']}")
        logger.info(f"   Query type      : {result.get('query_type', 'N/A')}")
        logger.info(f"   CRAG passed     : {result.get('crag_passed', 'N/A')}")
        logger.info(f"   Grounding score : {result.get('grounding_score', 0):.2f}")
        logger.info(f"   Retrieval tries : {result.get('retrieval_attempts', 'N/A')}")
        if result.get("calc_result"):
            logger.info(f"   Calculation     :\n{result['calc_result']}")
        logger.info(f"\n   ANSWER:\n{result.get('answer', 'ERROR')}")
        logger.info(f"\n   Agent steps ({len(result.get('steps', []))}):")
        for j, step in enumerate(result.get("steps", []), 1):
            logger.info(f"     {j}. [{step['node']}] {step['detail'][:80]}")
        logger.info(f"\n   Trace saved → {trace_path}")

        results.append(result)

    # Combined output
    combined = LOG_DIR / f"{ts}_all_test_results.json"
    with open(combined, "w") as f:
        json.dump({
            "run_timestamp": ts,
            "total"        : len(TEST_QUERIES),
            "passed"       : sum(1 for r in results if r["status"] == "success"),
            "results"      : results,
        }, f, indent=2, default=str)

    logger.info(f"\n╔══════════════════════════════════════════╗")
    logger.info(f"║  All tests complete                       ║")
    logger.info(f"║  Passed: {sum(1 for r in results if r['status']=='success')}/3                            ║")
    logger.info(f"║  Combined → {str(combined):<28} ║")
    logger.info(f"╚══════════════════════════════════════════╝")

    return results


if __name__ == "__main__":
    run_all_tests()
