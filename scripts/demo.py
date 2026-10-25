"""Demo script executing both evaluation requests against a running Document Agent service."""

import argparse
import json
from pathlib import Path
import sys
import httpx


def run_demo(base_url: str = "http://localhost:8000") -> None:
    # Ensure stdout/stderr handle UTF-8 cleanly
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    base_url = base_url.rstrip("/")
    samples_path = Path(__file__).resolve().parent.parent / "samples" / "test_inputs.json"
    demo_output_dir = Path(__file__).resolve().parent.parent / "demo_output"
    demo_output_dir.mkdir(parents=True, exist_ok=True)

    if not samples_path.exists():
        print(f"Error: test inputs file not found at {samples_path}", file=sys.stderr)
        sys.exit(1)

    with open(samples_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    client = httpx.Client(timeout=360.0)

    # 1. Health check
    print("=" * 80)
    print(f"  AUTONOMOUS DOCUMENT AGENT DEMO - Connecting to {base_url}")
    print("=" * 80)

    try:
        health_resp = client.get(f"{base_url}/health")
        if health_resp.status_code == 200:
            h = health_resp.json()
            print(f"Server Status: OK | Provider: {h.get('provider')} | Model: {h.get('model')}")
        else:
            print(f"Warning: Health check returned HTTP {health_resp.status_code}")
    except Exception as exc:
        print(f"Error connecting to {base_url}: {exc}", file=sys.stderr)
        print("Please ensure the FastAPI service is running: uvicorn app.main:app --reload", file=sys.stderr)
        sys.exit(1)

    for idx, tc in enumerate(test_cases, start=1):
        print("\n" + "#" * 80)
        print(f"  TEST CASE {idx}: {tc['name'].upper()}")
        print("#" * 80)
        print(f"\nUser Request:\n  \"{tc['request']}\"\n")
        print("Sending request to POST /agent ... (planning, executing, reflecting)")

        try:
            resp = client.post(
                f"{base_url}/agent",
                json={"request": tc["request"], "include_document_base64": False},
            )
        except Exception as exc:
            print(f"Failed to communicate with service: {exc}", file=sys.stderr)
            continue

        if resp.status_code != 200:
            print(f"Request failed with HTTP {resp.status_code}: {resp.text}")
            continue

        data = resp.json()

        print("\n" + "-" * 70)
        print(f"STATUS: {data.get('status').upper()} | Request ID: {data.get('request_id')}")
        print("-" * 70)
        print(f"Summary Message:\n  {data.get('message')}\n")

        plan = data.get("plan", {})
        print(f"Document Spec: '{plan.get('title')}' ({plan.get('doc_type')}) -> Audience: {plan.get('audience')}")

        # Assumptions
        assumptions = plan.get("assumptions", [])
        print(f"\nExplicit Assumptions ({len(assumptions)}):")
        for asm in assumptions:
            print(f"  * {asm}")

        # Open questions
        open_q = plan.get("open_questions", [])
        if open_q:
            print(f"\nOpen Questions ({len(open_q)}):")
            for q in open_q:
                print(f"  ? {q}")

        # Task list
        tasks = plan.get("tasks", [])
        print(f"\nAutonomous Task Execution List ({len(tasks)} tasks):")
        print(f"{'ID':<6} {'Title':<30} {'Tool':<20} {'Deps':<12} {'Status':<10} {'Attempts'}")
        print("-" * 88)
        for t in tasks:
            tool_str = t.get("tool") or "(LLM Write)"
            deps_str = ",".join(t.get("depends_on", [])) or "None"
            status_str = t.get("status", "unknown")
            if t.get("used_fallback"):
                status_str += " (fallback)"
            print(f"{t.get('id'):<6} {t.get('title')[:28]:<30} {tool_str:<20} {deps_str:<12} {status_str:<10} {t.get('attempts')}")

        # Reflection
        refl = data.get("reflection")
        if refl:
            print(f"\nMandatory Reflection Quality Gate:")
            print(f"  Initial Critic Score: {refl.get('initial_score')}/10  |  Final Critic Score: {refl.get('final_score')}/10")
            print(f"  Rounds: {refl.get('rounds')}  |  Issues Found: {refl.get('issues_found')}")
            if refl.get("revisions"):
                print("  Revisions Applied:")
                for rev in refl.get("revisions"):
                    print(f"    [OK] {rev}")
            if refl.get("remaining_issues"):
                print("  Remaining Issues:")
                for rem in refl.get("remaining_issues"):
                    print(f"    [!] {rem}")

        # Warnings
        warnings = data.get("warnings", [])
        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for w in warnings:
                print(f"  [WARN] {w}")

        # Metrics
        metrics = data.get("metrics", {})
        print(f"\nExecution Metrics:")
        print(f"  LLM Calls: {metrics.get('llm_calls')}  |  Duration: {metrics.get('elapsed_seconds')}s")

        # Download Document
        doc_info = data.get("document", {})
        download_url = doc_info.get("download_url")
        filename = doc_info.get("filename")

        if download_url and filename:
            target_path = demo_output_dir / filename
            doc_resp = client.get(f"{base_url}{download_url}")
            if doc_resp.status_code == 200:
                target_path.write_bytes(doc_resp.content)
                print(f"\nDownloaded Word Document: {target_path} ({len(doc_resp.content)} bytes)")
            else:
                print(f"\nFailed to download document: HTTP {doc_resp.status_code}")

    print("\n" + "=" * 80)
    print("  DEMO COMPLETE - All documents downloaded to demo_output/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Document Agent live demo.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Document Agent base URL")
    args = parser.parse_args()
    run_demo(base_url=args.base_url)
