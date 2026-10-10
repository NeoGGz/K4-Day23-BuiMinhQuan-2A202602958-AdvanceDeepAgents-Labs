"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


import re

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_URL_RE = re.compile(r"https?://[^\s)\]]+")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    seen_urls = set()
    sources_by_n = {}
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"invalid source entry (not a dict): {entry!r}")
            continue
        n = entry.get("n")
        if not isinstance(n, int):
            problems.append(f"source entry has non-integer n: {n!r}")
        elif n in sources_by_n:
            problems.append(f"duplicate source number n={n} in sources.json")
        else:
            sources_by_n[n] = entry

        url = entry.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source n={n!r} has invalid URL: {url!r}")
        elif url in seen_urls:
            problems.append(f"duplicate url in sources.json: {url}")
        else:
            seen_urls.add(url)

    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing '## References' heading")
        return problems

    ref_match = matches[-1]
    body = report_text[:ref_match.start()]
    refs_text = report_text[ref_match.end():]

    # Extract cited numbers from body (excluding code blocks)
    segments = _CODE.split(body)
    cited = set()
    for i, seg in enumerate(segments):
        if i % 2 == 1:
            continue
        for match in _GROUP.finditer(seg):
            for num in _group_numbers(match.group(1)):
                cited.add(num)

    source_ns = set(sources_by_n.keys())
    for num in sorted(cited):
        if num not in source_ns:
            problems.append(f"[{num}] cited but missing from sources.json")

    for num in sorted(source_ns):
        if num not in cited:
            problems.append(f"source [{num}] never cited")

    # References section
    ref_lines = [line.strip() for line in refs_text.splitlines() if line.strip()]
    seen_ref_ns = set()
    ref_line_by_n = {}

    for line in ref_lines:
        line_match = re.match(r"^\[(\d+)\]\s*(.*)", line)
        if not line_match:
            continue
        n = int(line_match.group(1))
        if n in seen_ref_ns:
            problems.append(f"duplicate reference line for [{n}]")
        else:
            seen_ref_ns.add(n)
            ref_line_by_n[n] = line

        if n not in source_ns:
            problems.append(f"reference [{n}] exists in References but is not in sources.json")
            continue

        # Check URL on this reference line
        urls = _URL_RE.findall(line)
        expected_url = sources_by_n[n].get("url")
        if len(urls) == 0:
            problems.append(f"reference line [{n}] has no URL")
        elif len(urls) > 1:
            problems.append(f"reference line [{n}] bundles multiple URLs: {urls}")
        elif urls[0] != expected_url:
            problems.append(f"reference line [{n}] URL {urls[0]} does not match sources.json URL {expected_url}")

    for num in sorted(source_ns):
        if num not in seen_ref_ns:
            problems.append(f"missing reference line for source [{num}]")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
