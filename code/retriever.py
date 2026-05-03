"""
retriever.py — Keyword-based article retriever for support triage.

Walks data/ folder, parses YAML-frontmattered .md files, scores them
against a query, and returns the top-K most relevant articles.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE.parent / "data"

# ---------------------------------------------------------------------------
# Stop-words (tiny set — keeps scoring fast without a library dep)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "with", "this", "that", "are", "was",
        "be", "by", "as", "from", "how", "do", "i", "my", "can", "not",
        "what", "when", "where", "why", "which", "who", "will", "have",
        "has", "had", "if", "so", "no", "yes", "get", "got", "its",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords & short tokens."""
    raw = re.split(r"[^a-z0-9]+", text.lower())
    return [t for t in raw if len(t) > 1 and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Article:
    path: Path
    company: str                     # e.g. "claude", "hackerrank", "visa"
    product_area: str                # e.g. "amazon-bedrock", "settings"
    title: str
    source_url: str
    breadcrumbs: list[str]
    body: str
    score: float = field(default=0.0, compare=False)

    # Pre-tokenised caches (computed once, used for scoring)
    _title_tokens: list[str] = field(default_factory=list, repr=False)
    _body_tokens: list[str] = field(default_factory=list, repr=False)
    _breadcrumb_tokens: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._title_tokens = _tokenize(self.title)
        self._body_tokens = _tokenize(self.body)
        bc_text = " ".join(self.breadcrumbs)
        self._breadcrumb_tokens = _tokenize(bc_text)

    @property
    def snippet(self) -> str:
        """First 300 chars of body, stripped of markdown symbols."""
        clean = re.sub(r"[#*_`>\-]+", " ", self.body)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean[:300]


# ---------------------------------------------------------------------------
# Frontmatter parser (no PyYAML dependency)
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Split YAML frontmatter from body.

    Returns (metadata_dict, body_text).  Handles missing frontmatter
    gracefully by returning an empty dict.
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    yaml_block = text[3:end].strip()
    body = text[end + 4:].lstrip("\n")

    meta: dict = {}
    current_key: Optional[str] = None
    list_items: list[str] = []
    in_list = False

    for line in yaml_block.splitlines():
        list_match = re.match(r"^  - (.+)$", line)
        kv_match = re.match(r'^([a-z_]+):\s*"?([^"]*)"?\s*$', line)
        list_key_match = re.match(r'^([a-z_]+):\s*$', line)

        if list_match:
            list_items.append(list_match.group(1).strip())
        elif list_key_match:
            if current_key and in_list:
                meta[current_key] = list_items
            current_key = list_key_match.group(1)
            list_items = []
            in_list = True
        elif kv_match:
            if current_key and in_list:
                meta[current_key] = list_items
                in_list = False
            current_key = kv_match.group(1)
            meta[current_key] = kv_match.group(2).strip()
            in_list = False
        # lines that don't match (e.g. multi-line values) are skipped

    if current_key and in_list:
        meta[current_key] = list_items

    return meta, body


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _derive_company_and_area(path: Path, data_dir: Path) -> tuple[str, str]:
    """
    Given a .md file path and the data/ root, return (company, product_area).

    e.g.  data/claude/amazon-bedrock/file.md  → ("claude", "amazon-bedrock")
          data/claude/mobile/ios/file.md       → ("claude", "mobile")
          data/visa/support.md                 → ("visa", "")
    """
    try:
        rel = path.relative_to(data_dir)
    except ValueError:
        return ("unknown", "")

    parts = rel.parts  # ('claude', 'amazon-bedrock', 'file.md') etc.
    company = parts[0] if len(parts) >= 1 else "unknown"
    # product_area is the second path component when it is a directory segment
    # (i.e. not the file itself)
    if len(parts) >= 3:
        product_area = parts[1]
    elif len(parts) == 2 and not parts[1].endswith(".md"):
        product_area = parts[1]
    else:
        product_area = ""

    return company, product_area


def load_articles(data_dir: Path = DATA_DIR) -> list[Article]:
    """Walk data_dir, parse every .md, return a list of Article objects."""
    articles: list[Article] = []

    for md_path in sorted(data_dir.rglob("*.md")):
        try:
            raw = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        meta, body = _parse_frontmatter(raw)

        title = meta.get("title", md_path.stem.replace("-", " ").title())
        source_url = meta.get("source_url", "")
        breadcrumbs = meta.get("breadcrumbs", [])
        if isinstance(breadcrumbs, str):
            breadcrumbs = [breadcrumbs]

        company, product_area = _derive_company_and_area(md_path, data_dir)

        articles.append(
            Article(
                path=md_path,
                company=company,
                product_area=product_area,
                title=title,
                source_url=source_url,
                breadcrumbs=breadcrumbs,
                body=body,
            )
        )

    return articles


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_BOOST_COMPANY      = 0.50   # query explicitly names the company
_BOOST_PRODUCT_AREA = 0.20   # query contains a product-area token
_BOOST_TITLE        = 0.40   # title tokens overlap well with query
_BOOST_BREADCRUMB   = 0.25   # breadcrumb overlap


def _overlap_ratio(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """Fraction of query tokens found in doc_tokens (token overlap recall)."""
    if not query_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    hits = sum(1 for t in query_tokens if t in doc_set)
    return hits / len(query_tokens)


def score_article(article: Article, query_tokens: list[str], company_hint: str = "") -> float:
    """
    Return a relevance score in [0, ∞).

    Components:
      • body overlap   — fraction of query tokens found in the article body
      • title boost    — extra credit when the title covers the query well
      • breadcrumb boost — extra credit when breadcrumbs match query tokens
      • company boost  — extra credit when query or hint names this company
      • product_area boost — extra credit when product-area tokens appear in query
    """
    body_score = _overlap_ratio(query_tokens, article._body_tokens)
    title_boost = _overlap_ratio(query_tokens, article._title_tokens) * _BOOST_TITLE
    bc_boost = _overlap_ratio(query_tokens, article._breadcrumb_tokens) * _BOOST_BREADCRUMB

    # Company boost: check if company name appears in query tokens or hint
    query_set = set(query_tokens)
    company_tokens = _tokenize(article.company)
    company_boost = 0.0
    if any(t in query_set for t in company_tokens) or (
        company_hint and article.company.lower() == company_hint.lower()
    ):
        company_boost = _BOOST_COMPANY

    # Product-area boost
    area_tokens = _tokenize(article.product_area.replace("-", " "))
    area_boost = 0.0
    if area_tokens and any(t in query_set for t in area_tokens):
        area_boost = _BOOST_PRODUCT_AREA

    return body_score + title_boost + bc_boost + company_boost + area_boost


def retrieve(
    query: str,
    articles: list[Article],
    top_k: int = 5,
    company_hint: str = "",
) -> list[Article]:
    """
    Score every article against `query` and return the top-K results,
    highest score first.  Articles with score == 0 are omitted.
    """
    query_tokens = _tokenize(query)
    scored: list[Article] = []

    for article in articles:
        s = score_article(article, query_tokens, company_hint=company_hint)
        if s > 0:
            # Clone score into the dataclass field (non-destructive for the cache)
            article.score = s
            scored.append(article)

    scored.sort(key=lambda a: a.score, reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Prompt formatter
# ---------------------------------------------------------------------------

def format_for_prompt(articles: list[Article], max_body_chars: int = 800) -> str:
    """
    Format a list of retrieved articles as clean, prompt-ready text.

    Each article is rendered with its rank, metadata, and a body excerpt.
    """
    if not articles:
        return "No relevant articles found."

    lines: list[str] = []
    for i, art in enumerate(articles, start=1):
        breadcrumb_str = " > ".join(art.breadcrumbs) if art.breadcrumbs else art.product_area
        body_excerpt = art.snippet[:max_body_chars]

        lines.append(f"[{i}] {art.title}")
        lines.append(f"    Company: {art.company}  |  Area: {art.product_area}")
        if breadcrumb_str:
            lines.append(f"    Path: {breadcrumb_str}")
        if art.source_url:
            lines.append(f"    URL: {art.source_url}")
        lines.append(f"    Score: {art.score:.3f}")
        lines.append("")
        lines.append(body_excerpt)
        lines.append("")
        lines.append("-" * 72)
        lines.append("")

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# __main__ — quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "how do I cancel my Claude Pro subscription on iOS"

    print(f"Loading articles from {DATA_DIR} …")
    articles = load_articles()
    print(f"Loaded {len(articles)} articles.\n")

    print(f"Query: {query!r}\n")
    results = retrieve(query, articles, top_k=5)

    if not results:
        print("No results found.")
    else:
        print(format_for_prompt(results))
        print(f"\nTop result: '{results[0].title}'  (score={results[0].score:.3f})")
