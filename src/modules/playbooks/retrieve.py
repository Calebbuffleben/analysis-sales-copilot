"""Local TF-IDF retrieval over cached playbook templates (Live hot-path safe).

Ceiling: bag-of-words TF-IDF in process memory for tens of templates.
Upgrade path: SBert embeddings if Live ever preloads the model.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

_TOKEN_RE = re.compile(r'[a-zA-ZÀ-ÿ0-9_]+', re.UNICODE)

# Enable retrieve / ranked catalog when tenant has more than this many templates.
RETRIEVE_MIN_TEMPLATES = 12
CATALOG_PROMPT_MAX = 20


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or '') if t]


def _template_doc(template: dict[str, Any]) -> str:
    """Indexable text: key, title, description, step labels, PDF excerpt — not payloads."""
    parts: list[str] = [
        str(template.get('key') or ''),
        str(template.get('title') or ''),
        str(template.get('description') or ''),
        str(template.get('sourceTextExcerpt') or ''),
    ]
    # Underscores in keys are useful as separate tokens (preco_vs_concorrente).
    key = str(template.get('key') or '')
    parts.append(key.replace('_', ' '))
    steps = template.get('steps')
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            label = step.get('label')
            if label is not None:
                parts.append(str(label))
            detail = step.get('detail')
            if detail is not None:
                parts.append(str(detail)[:120])
    return ' '.join(parts)


@dataclass(frozen=True)
class RetrieveHit:
    key: str
    title: str
    score: float
    template: dict[str, Any]


class PlaybookIndex:
    """In-memory TF-IDF index over playbook templates."""

    def __init__(
        self,
        templates: list[dict[str, Any]],
        *,
        vectors: list[dict[str, float]],
        idf: dict[str, float],
    ) -> None:
        self._templates = templates
        self._vectors = vectors
        self._idf = idf

    @classmethod
    def from_templates(cls, templates: list[dict[str, Any]]) -> PlaybookIndex:
        docs: list[list[str]] = []
        kept: list[dict[str, Any]] = []
        for t in templates:
            if not isinstance(t, dict):
                continue
            key = str(t.get('key') or '').strip()
            if not key:
                continue
            kept.append(t)
            docs.append(_tokenize(_template_doc(t)))

        n = len(docs)
        df: Counter[str] = Counter()
        for tokens in docs:
            df.update(set(tokens))
        idf: dict[str, float] = {}
        for term, count in df.items():
            idf[term] = math.log((1.0 + n) / (1.0 + count)) + 1.0

        vectors: list[dict[str, float]] = []
        for tokens in docs:
            tf = Counter(tokens)
            length = max(sum(tf.values()), 1)
            vec: dict[str, float] = {}
            for term, freq in tf.items():
                vec[term] = (freq / length) * idf.get(term, 0.0)
            vectors.append(_l2_normalize(vec))

        return cls(kept, vectors=vectors, idf=idf)

    @property
    def size(self) -> int:
        return len(self._templates)

    def retrieve(self, query: str, *, k: int = 3) -> list[RetrieveHit]:
        q = (query or '').strip()
        if not q or k <= 0 or not self._templates:
            return []
        q_tokens = _tokenize(q)
        if not q_tokens:
            return []
        tf = Counter(q_tokens)
        length = max(sum(tf.values()), 1)
        q_vec: dict[str, float] = {}
        for term, freq in tf.items():
            q_vec[term] = (freq / length) * self._idf.get(term, 0.0)
        q_vec = _l2_normalize(q_vec)
        if not q_vec:
            return []

        scored: list[tuple[float, int]] = []
        for i, doc_vec in enumerate(self._vectors):
            score = _cosine(q_vec, doc_vec)
            if score > 0.0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], str(self._templates[x[1]].get('key') or '')))
        hits: list[RetrieveHit] = []
        for score, i in scored[:k]:
            t = self._templates[i]
            hits.append(
                RetrieveHit(
                    key=str(t.get('key') or ''),
                    title=str(t.get('title') or ''),
                    score=score,
                    template=t,
                ),
            )
        return hits

    def top_templates_for_prompt(
        self,
        query: str,
        *,
        max_items: int = CATALOG_PROMPT_MAX,
    ) -> list[dict[str, Any]]:
        """Rank templates for system_instruction; empty query → stable key order."""
        if not self._templates:
            return []
        q = (query or '').strip()
        if q:
            hits = self.retrieve(q, k=max_items)
            if hits:
                return [h.template for h in hits]
        # Stable fallback: sort by key (documented tie-break).
        ordered = sorted(
            self._templates,
            key=lambda t: str(t.get('key') or ''),
        )
        return ordered[:max_items]


def format_retrieve_nudge(hits: list[RetrieveHit]) -> str:
    if not hits:
        return ''
    lines = [
        'Candidatos de playbook (no máximo um playbook_template_key se aplicável):',
    ]
    for h in hits:
        title = (h.title or '').strip()[:80]
        lines.append(f'- {h.key}: {title}' if title else f'- {h.key}')
    return '\n'.join(lines)


def build_retrieve_query(
    *,
    context_summary: str = '',
    retrieve_query_hint: str = '',
) -> str:
    parts = [
        (context_summary or '').strip(),
        (retrieve_query_hint or '').strip(),
    ]
    return ' '.join(p for p in parts if p)


def hint_from_emit_feedback_args(args: dict[str, Any]) -> str:
    """Short retrieval hint stored after each tool call (no I/O)."""
    parts: list[str] = []
    ft = args.get('feedback_type')
    if ft:
        parts.append(str(ft))
    evidence = args.get('evidence_text')
    if evidence:
        parts.append(str(evidence)[:200])
    feedback = args.get('feedback')
    if feedback:
        parts.append(str(feedback)[:160])
    estado = args.get('estado')
    if isinstance(estado, dict):
        for key in (
            'objecoes_detectadas',
            'product',
            'pain_points',
            'objections',
            'fase_spin',
        ):
            val = estado.get(key)
            if val is None or val == '' or val == []:
                continue
            if isinstance(val, list):
                parts.append(' '.join(str(x) for x in val[:8]))
            else:
                parts.append(str(val)[:120])
    return ' '.join(parts)[:500]


def _l2_normalize(vec: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm <= 0.0:
        return {}
    return {k: v / norm for k, v in vec.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())
