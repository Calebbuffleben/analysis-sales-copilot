"""Pure playbook hint parse, {{var}} interpolate, and resolve (Nest mirror)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PLAYBOOK_MAX_STEPS = 5
PLAYBOOK_MAX_STEP_LABEL_CHARS = 120
PLAYBOOK_MAX_STEP_DETAIL_CHARS = 280
PLAYBOOK_MAX_ACTION_PAYLOAD_CHARS = 2000
PLAYBOOK_MAX_TEMPLATE_KEY_CHARS = 64
PLAYBOOK_MAX_STEP_ID_CHARS = 64
PLAYBOOK_MAX_TITLE_CHARS = 160

_PLACEHOLDER_RE = re.compile(r'\{\{([a-zA-Z0-9_]+)\}\}')


def parse_playbook_url_allowlist_env(raw: Optional[str]) -> set[str]:
    s = (raw or '').strip()
    if not s:
        return set()
    return {h.strip().lower() for h in s.split(',') if h.strip()}


def interpolate_playbook_placeholders(
    input_str: str,
    variables: dict[str, str],
) -> str:
    return _PLACEHOLDER_RE.sub(
        lambda m: variables.get(m.group(1), ''),
        input_str,
    )


def _hostname_allowed(hostname: str, allowlist: set[str]) -> bool:
    h = hostname.lower()
    if h in allowlist:
        return True
    for base in allowlist:
        if h.endswith('.' + base):
            return True
    return False


def is_https_url_allowed_for_playbook(url_string: str, allowlist: set[str]) -> bool:
    if not allowlist:
        return False
    try:
        u = urlparse(url_string)
        if u.scheme != 'https' or not u.hostname:
            return False
        return _hostname_allowed(u.hostname, allowlist)
    except Exception:
        return False


def parse_playbook_hint_json(raw: Optional[str]) -> Optional[dict[str, Any]]:
    trimmed = (raw or '').strip()
    if not trimmed:
        return None
    try:
        obj = json.loads(trimmed)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    key_raw = (
        obj.get('playbook_template_key')
        or obj.get('template_key')
        or obj.get('playbookTemplateKey')
    )
    template_key = key_raw.strip() if isinstance(key_raw, str) else ''
    if not template_key:
        return None
    vars_raw = (
        obj.get('playbook_variables')
        or obj.get('playbookVariables')
        or obj.get('variables')
        or {}
    )
    variables: dict[str, str] = {}
    if isinstance(vars_raw, dict):
        for k, v in vars_raw.items():
            if isinstance(v, str):
                variables[str(k)] = v
            elif v is not None:
                variables[str(k)] = str(v)
    return {'templateKey': template_key, 'variables': variables}


def _truncate(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len]


def normalize_template_key(key: str) -> Optional[str]:
    t = key.strip()
    if not t:
        return None
    return _truncate(t, PLAYBOOK_MAX_TEMPLATE_KEY_CHARS)


def format_catalog_for_prompt(
    templates: list[dict[str, Any]],
    *,
    max_items: int = 20,
) -> str:
    """Compact catalog lines for Live system_instruction."""
    lines: list[str] = []
    for t in templates[:max_items]:
        key = str(t.get('key') or '').strip()
        if not key:
            continue
        title = _truncate(str(t.get('title') or '').strip(), 80)
        line = f'- {key}: {title}' if title else f'- {key}'
        excerpt = str(t.get('sourceTextExcerpt') or '').strip()
        if excerpt:
            # Cap keeps system_instruction small (latency / token budget).
            line = f'{line} | {_truncate(excerpt, 160)}'
        lines.append(line)
    if not lines:
        return ''
    return (
        'Playbooks do tenant (use playbook_template_key exatamente):\n'
        + '\n'.join(lines)
    )


def resolve_playbook_metadata(
    *,
    templates_by_key: dict[str, dict[str, Any]],
    playbook_hint_json: Optional[str],
    url_allowlist: set[str],
) -> Optional[dict[str, Any]]:
    """Build FeedbackPlaybookMetadata dict or None. No I/O."""
    hint = parse_playbook_hint_json(playbook_hint_json)
    if not hint:
        return None
    normalized = normalize_template_key(str(hint['templateKey']))
    if not normalized:
        return None
    template = templates_by_key.get(normalized)
    if template is None:
        # try case-sensitive exact then lower match
        template = templates_by_key.get(normalized.lower())
        if template is None:
            for k, v in templates_by_key.items():
                if k.lower() == normalized.lower():
                    template = v
                    normalized = k
                    break
    if template is None:
        return None

    vars_ = hint.get('variables') or {}
    if not isinstance(vars_, dict):
        vars_ = {}
    title_raw = interpolate_playbook_placeholders(
        str(template.get('title') or ''),
        vars_,
    )
    title = _truncate(title_raw, PLAYBOOK_MAX_TITLE_CHARS)
    steps_raw = template.get('steps')
    if not isinstance(steps_raw, list):
        steps_raw = []

    steps: list[dict[str, Any]] = []
    for raw in steps_raw:
        if len(steps) >= PLAYBOOK_MAX_STEPS:
            break
        resolved = _resolve_step(raw, vars_, url_allowlist)
        if resolved:
            steps.append(resolved)
    if not steps:
        return None

    out: dict[str, Any] = {'templateKey': normalized, 'steps': steps}
    if title:
        out['title'] = title
    return out


def _resolve_step(
    raw: Any,
    vars_: dict[str, str],
    url_allowlist: set[str],
) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    id_src = raw.get('id')
    label_src = raw.get('label')
    id_s = str(id_src).strip() if id_src is not None else ''
    label_s = str(label_src).strip() if label_src is not None else ''
    step_id = _truncate(
        interpolate_playbook_placeholders(id_s, vars_),
        PLAYBOOK_MAX_STEP_ID_CHARS,
    )
    label = _truncate(
        interpolate_playbook_placeholders(label_s, vars_),
        PLAYBOOK_MAX_STEP_LABEL_CHARS,
    )
    if not step_id or not label:
        return None

    detail: Optional[str] = None
    if raw.get('detail') is not None and raw.get('detail') != '':
        d = str(raw.get('detail')).strip()
        di = _truncate(
            interpolate_playbook_placeholders(d, vars_),
            PLAYBOOK_MAX_STEP_DETAIL_CHARS,
        )
        if di:
            detail = di

    action_raw = raw.get('action')
    if not isinstance(action_raw, dict):
        return None
    type_str = str(action_raw.get('type') or '').strip().lower()
    payload_src = action_raw.get('payload')
    payload_s = '' if payload_src is None else str(payload_src)
    payload = _truncate(
        interpolate_playbook_placeholders(payload_s, vars_),
        PLAYBOOK_MAX_ACTION_PAYLOAD_CHARS,
    )
    if type_str not in {'copy_text', 'open_url', 'noop'}:
        return None
    safe = _sanitize_action(type_str, payload, url_allowlist)
    if safe is None:
        return None
    step: dict[str, Any] = {
        'id': step_id,
        'label': label,
        'action': safe,
    }
    if detail is not None:
        step['detail'] = detail
    return step


def _sanitize_action(
    action_type: str,
    payload: str,
    url_allowlist: set[str],
) -> Optional[dict[str, str]]:
    if action_type == 'noop':
        return {'type': 'noop', 'payload': ''}
    if action_type == 'copy_text':
        if not payload.strip():
            return None
        return {'type': 'copy_text', 'payload': payload}
    if action_type == 'open_url':
        if not is_https_url_allowed_for_playbook(payload, url_allowlist):
            logger.warning(
                'Playbook open_url rejected | url=%s',
                payload[:120],
            )
            return None
        return {'type': 'open_url', 'payload': payload}
    return None
