import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = (
    PROJECT_ROOT
    / "data"
    / "runs"
    / "d9583a96dc5244698797d92b41bff103.json"
)

SOURCE_ID = "pg1232:chap09:p0009"

CASES = [
    {
        "case_id": "review_supported",
        "claim": (
            "Government through magistrates can leave a prince "
            "insecure in troubled times because the officials "
            "may undermine his government."
        ),
        "expected_verdict": "supported",
    },
    {
        "case_id": "review_overextended",
        "claim": (
            "Government through magistrates or noble intermediaries "
            "can leave a prince insecure in troubled times because "
            "these officials may undermine his government."
        ),
        "expected_verdict": "partly_supported",
    },
    {
        "case_id": "review_contradicted",
        "claim": (
            "Government through magistrates guarantees a prince "
            "security in troubled times because those officials "
            "cannot undermine his government."
        ),
        "expected_verdict": "unsupported",
    },
]


def main():
    original_run = json.loads(
        RUN_PATH.read_text(encoding="utf-8")
    )

    source = next(
        source
        for source in original_run["evidence"]
        if source["source_id"] == SOURCE_ID
    )

    output_dir = PROJECT_ROOT / "data" / "review_cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        # Match the input format expected by review_run.py.
        fixture = {
            "run_id": case["case_id"],
            "record_type": "authored_review_test",
            "evidence": [source],
            "answer": {
                "claims": [{
                    "text": case["claim"],
                    "evidence_type": "paraphrase",
                    "source_ids": [SOURCE_ID],
                }],
            },
        }

        output_path = output_dir / f"{case['case_id']}.json"
        output_path.write_text(
            json.dumps(fixture, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"{case['case_id']}: "
            f"expected {case['expected_verdict']}"
        )


if __name__ == "__main__":
    main()