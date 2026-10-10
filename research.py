"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    cleaned = re.sub(r"[^\w]+", "-", str(topic).strip().lower()).strip("-")
    trimmed = cleaned[:60].strip("-")
    return trimmed if trimmed else "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Conduct a comprehensive, multi-agent academic research survey on the topic: '{topic}'.\n\n"
        f"Follow the mandated research protocol:\n"
        f"1. Plan using `write_todos` and split '{topic}' into N >= 3 independent, distinct sub-questions.\n"
        f"2. Delegate each sub-question to the `researcher` subagent in parallel using the `task` tool. "
        f"Ensure each delegation specifies the topic, sub-question, destination note file in {WORKDIR}/research/notes/, "
        f"and instructs querying at least 2 different source families.\n"
        f"3. Check the researcher outputs and inspect the notes in {WORKDIR}/research/notes/.\n"
        f"4. Consolidate all sources into {SOURCES_PATH} (JSON list with keys n, id, url, title, date, source), "
        f"numbered sequentially from 1 with NO duplicate URLs. Ensure sources cover at least 3 of the 4 source families "
        f"(arxiv, hf-daily, hf-search, web). Delegate additional search if fewer than 3 families are found.\n"
        f"5. Write the comprehensive survey body to {REPORT_PATH} following REPORT_TEMPLATE.md (TL;DR, Background, 3-6 thematic sections, "
        f"Trends and open problems). Do NOT write a '## References' section manually. Cite facts inline using [n]. "
        f"Ensure citations reference at least 3 source families.\n"
        f"6. Execute {FINALIZER_PATH} to automatically generate ## References and clean/renumber {SOURCES_PATH}.\n"
        f"7. Execute {VALIDATOR_PATH} to verify all citations resolve, fixing any issues until it outputs OK.\n"
        f"8. Delegate 2-3 key claims and URLs to `citation-checker` using `task` to spot-check accuracy."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_counts = Counter()
    subagent_calls = 0
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        # Tool calls
        tc = getattr(msg, "tool_calls", None)
        if tc:
            for call in tc:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name:
                    tool_counts[name] += 1
                    if name == "task":
                        subagent_calls += 1

        # Token usage
        usage = getattr(msg, "usage_metadata", None)
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
        elif hasattr(msg, "response_metadata"):
            resp_meta = getattr(msg, "response_metadata", {})
            token_usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
            if isinstance(token_usage, dict):
                input_tokens += int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
                output_tokens += int(token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0)

    return {
        "model": str(model_name),
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_counts),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("Report file is missing or empty in sandbox")

    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError("sources.json is missing or empty in sandbox")

    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources, list):
            raise ValueError("sources.json must be a JSON array")
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}") from exc

    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    summary = summarize(messages, elapsed, model_name)

    source_families = sorted({s.get("source") for s in sources if isinstance(s, dict) and s.get("source")})

    meta = {
        "topic": topic,
        **summary,
        "n_sources": len(sources),
        "source_families": source_families,
    }

    md_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"

    sources_path.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_bytes(report_bytes)

    return md_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    if not topic or not topic.strip():
        print("Usage: python research.py \"<topic>\"", file=sys.stderr)
        return 2

    topic = topic.strip()
    model = make_model()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "unknown")
    start = time.monotonic()

    try:
        with open_sandbox() as backend:
            res = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if res.exit_code != 0:
                raise RuntimeError(f"Failed to create directories in sandbox: {res.output}")

            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })

            agent = build_lead_agent(backend, model)
            result = None
            for invoke_attempt in range(3):
                try:
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                        config={"recursion_limit": 1000},
                    )
                    break
                except Exception as exc:
                    err_str = str(exc)
                    if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and invoke_attempt < 2:
                        print(f"Rate limited (429/RESOURCE_EXHAUSTED), pausing 35s before retry (attempt {invoke_attempt + 1}/3)...")
                        time.sleep(35)
                        continue
                    raise

            elapsed = time.monotonic() - start
            messages = result.get("messages", []) if isinstance(result, dict) else []
            report_file = save_outputs(backend, topic, messages, elapsed, model_name)
            print(f"SUCCESS: Report saved to {report_file}")
            return 0
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAILED: Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
