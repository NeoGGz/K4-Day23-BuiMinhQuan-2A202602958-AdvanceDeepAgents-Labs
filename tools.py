"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


import random
import re

_LAST_ARXIV_CALL = 0.0


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            retry_after = None
            is_retryable = False

            if isinstance(exc, RetryableError):
                is_retryable = True
                retry_after = exc.retry_after
            elif isinstance(exc, httpx.HTTPStatusError):
                if exc.response.status_code in {429, 500, 502, 503, 504}:
                    is_retryable = True
                    header_val = exc.response.headers.get("retry-after")
                    if header_val:
                        try:
                            retry_after = float(header_val)
                        except ValueError:
                            pass
            elif isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
                is_retryable = True

            if not is_retryable or attempt == attempts - 1:
                raise

            if retry_after is not None:
                delay = float(retry_after)
            else:
                delay = base * (2 ** attempt) + random.uniform(0.1, 1.0)
            delay = min(delay, cap)
            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        terms = re.findall(r"[\w-]+", query)
        if not terms:
            return "NO RESULTS"

        # Respect arXiv etiquette: at least 3 seconds between calls
        now = time.monotonic()
        elapsed = now - _LAST_ARXIV_CALL
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        _LAST_ARXIV_CALL = time.monotonic()

        search_query = " AND ".join(f"all:{t}" for t in terms)
        clamped_max = max(1, min(int(max_results), 30))
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": clamped_max,
        }

        def call():
            resp = httpx.get(ARXIV_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp.text

        xml_text = with_retry(call, attempts=6, cap=60.0)
        root = xml.etree.ElementTree.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            id_elem = entry.find("atom:id", ns)
            if id_elem is None or not id_elem.text:
                continue
            raw_id = id_elem.text.strip().split("/abs/")[-1]
            clean_id = re.sub(r"v\d+$", "", raw_id)
            url = f"https://arxiv.org/abs/{clean_id}"

            pub_elem = entry.find("atom:published", ns)
            published = pub_elem.text.strip()[:10] if pub_elem is not None and pub_elem.text else ""

            title_elem = entry.find("atom:title", ns)
            title = " ".join((title_elem.text or "").split()) if title_elem is not None else ""

            sum_elem = entry.find("atom:summary", ns)
            summary = " ".join((sum_elem.text or "").split())[:600] if sum_elem is not None else ""

            records.append({
                "id": clean_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(int(limit), 100))
        params = {"limit": clamped_limit}
        if date and date.strip():
            params["date"] = date.strip()

        def call():
            resp = httpx.get(HF_DAILY_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp.json()

        data = with_retry(call, attempts=5, cap=30.0)
        if not isinstance(data, list) or not data:
            return "NO RESULTS"

        records = []
        kw = keyword.strip().lower()
        for item in data:
            paper = item.get("paper", {})
            pid = paper.get("id")
            if not pid:
                continue

            title = " ".join(str(paper.get("title") or item.get("title") or "").split())
            summary = " ".join(str(paper.get("summary") or item.get("summary") or "").split())[:600]

            if kw and (kw not in title.lower() and kw not in summary.lower()):
                continue

            pub = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            records.append({
                "id": str(pid),
                "url": f"https://huggingface.co/papers/{pid}",
                "published": pub,
                "title": title,
                "summary": summary,
                "upvotes": paper.get("upvotes") or 0,
                "github": paper.get("githubRepo"),
                "stars": paper.get("githubStars") or 0,
            })

        records.sort(key=lambda r: r.get("upvotes") or 0, reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        clamped_limit = max(1, min(int(limit), 50))
        params = {"q": query, "limit": clamped_limit}

        def call():
            resp = httpx.get(HF_SEARCH_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp.json()

        data = with_retry(call, attempts=5, cap=30.0)
        if not isinstance(data, list) or not data:
            return "NO RESULTS"

        records = []
        for item in data:
            paper = item.get("paper", {})
            pid = paper.get("id")
            if not pid:
                continue

            title = " ".join(str(paper.get("title") or item.get("title") or "").split())
            raw_summary = paper.get("ai_summary") or paper.get("summary") or item.get("summary") or ""
            summary = " ".join(str(raw_summary).split())[:600]
            pub = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]

            records.append({
                "id": str(pid),
                "url": f"https://huggingface.co/papers/{pid}",
                "published": pub,
                "title": title,
                "summary": summary,
                "upvotes": paper.get("upvotes") or 0,
                "github": paper.get("githubRepo"),
                "stars": paper.get("githubStars") or 0,
            })

        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    exa_key = (os.getenv("EXA_API_KEY") or "").strip()
    endpoint = f"{EXA_URL}?exaApiKey={exa_key}" if exa_key else EXA_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def call():
        resp = httpx.post(endpoint, headers=headers, json=body, timeout=45.0)
        resp.raise_for_status()

        # Parse SSE or JSON response
        data = None
        for line in resp.text.splitlines():
            line_str = line.strip()
            if line_str.startswith("data:"):
                payload_str = line_str[5:].strip()
                if payload_str:
                    data = json.loads(payload_str)
                    break
        if data is None:
            data = resp.json()

        if "error" in data:
            err = data["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            if "rate limit" in err_msg.lower() or "too many" in err_msg.lower():
                raise RetryableError(f"Exa rate limited: {err_msg}", retry_after=20.0)
            raise RuntimeError(f"Exa RPC error: {err_msg}")

        result = data.get("result", {})
        meta = result.get("_meta")
        if meta:
            meta_str = str(meta).lower()
            if "ratelimit" in meta_str or "rate limit" in meta_str or "exceeded" in meta_str:
                raise RetryableError(f"Exa rate limited (_meta): {meta}", retry_after=20.0)

        contents = result.get("content", [])
        texts = [c.get("text", "") for c in contents if isinstance(c, dict) and c.get("type") == "text"]
        combined_text = "\n\n".join(texts).strip()

        # Check if text itself says rate limited
        lower_comb = combined_text.lower()
        if "rate limit" in lower_comb and ("exceeded" in lower_comb or "free tier" in lower_comb or "too many requests" in lower_comb):
            raise RetryableError(f"Exa rate limited in content: {combined_text[:100]}", retry_after=20.0)

        return combined_text

    return with_retry(call, attempts=5, base=2.0, cap=60.0)


def _redact_exa_key(msg: str) -> str:
    exa_key = (os.getenv("EXA_API_KEY") or "").strip()
    if exa_key:
        msg = msg.replace(exa_key, "[REDACTED]")
    return msg


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        obj = objective.strip() if objective and objective.strip() else f"find relevant research papers and surveys on {query}"
        clamped_num = max(1, min(int(num_results), 20))
        arguments = {
            "query": query,
            "objective": obj,
            "numResults": clamped_num,
        }
        res = _call_exa_mcp("web_search_exa", arguments)
        return res if res.strip() else "NO RESULTS"
    except Exception as exc:
        return _redact_exa_key(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        arguments = {"urls": [url]}
        res = _call_exa_mcp("web_fetch_exa", arguments)
        if not res.strip():
            return "NO RESULTS"
        return res[:12000]
    except Exception as exc:
        return _redact_exa_key(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
