"""Gemini-based semantic analyzer and state generator."""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

from .gemini_transport import (
    GeminiTransportMode,
    generate_with_transport_chain,
    is_auth_error_message,
    key_prefix,
)
from ...pipeline_latency import (
    LatencyTraceContext,
    log_gemini_prompt_sent,
    log_gemini_response_received,
)
from .llm_state_validator import (
    validate_llm_response,
)

logger = logging.getLogger(__name__)


class InvalidGeminiApiKeyError(Exception):
    """Raised when the configured Gemini API key is rejected by the API."""


class QuotaExhaustedError(Exception):
    """Raised when Gemini API quota is exhausted and we need to back off.
    
    This allows the fallback chain to distinguish between:
    - Temporary errors (network, timeout) → retry
    - Quota exhaustion (429) → use rule-based fallback immediately
    """
    pass


def uses_vertex_express_api(api_key: str) -> bool:
    """Return True for Vertex AI Express keys (``AQ.…`` prefix from AI Studio)."""
    return (api_key or '').strip().startswith('AQ.')


# Backwards-compatible alias used by tests and callers.
uses_rest_developer_api = uses_vertex_express_api


class GeminiAnalyzer:
    """Analyze transcription texts and manage conversation state using Gemini Flash.
    
    Features:
    - Quota protection: backs off on 429 errors to avoid hammering the API
    - Graceful degradation: returns empty fallback instead of crashing
    - Response validation: ensures LLM output matches expected schema
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = 'gemini-2.5-flash',
        client: Optional[Any] = None,
        slot_index: Optional[int] = None,
    ):
        self._api_key = (api_key or '').strip()
        self.model_name = model_name
        self._slot_index = slot_index
        self._api_key_prefix = key_prefix(self._api_key)
        self._cached_transport: Optional[GeminiTransportMode] = None
        self.client = client

        if client is not None:
            try:
                from google.genai import types

                self._generation_config_factory = types.GenerateContentConfig
            except Exception as exc:
                raise RuntimeError(
                    'google-genai is required for Gemini analysis. '
                    'Install python-service requirements before starting.',
                ) from exc
        else:
            if not self._api_key:
                logger.warning(
                    'No Gemini API key provided. Analysis might fail if not injected properly.',
                )
            self._generation_config_factory = None

        # Quota protection: track consecutive 429 errors
        self._consecutive_429_errors = 0
        self._backoff_until_ms = 0  # Don't call API until this timestamp

    def analyze(
        self,
        text: str,
        conversation_state: Dict[str, Any],
        speaker_role: str = 'client',
        latency_context: Optional[LatencyTraceContext] = None,
    ) -> Dict[str, Any]:
        """
        Send the transcribed text and current conversational state to Gemini.
        Returns a dict containing 'direct_feedback' (str), 'confidence' (float),
        'feedback_type' (str or None), and 'conversation_state' (dict).
        
        Features quota protection:
        - If in backoff period (429 errors), returns empty fallback immediately
        - Resets backoff counter on successful calls
        - Exponential backoff on consecutive 429 errors
        """
        # Check if we're in a backoff period due to quota exhaustion
        now_ms = int(time.time() * 1000)
        if now_ms < self._backoff_until_ms:
            remaining_sec = (self._backoff_until_ms - now_ms) / 1000
            logger.warning(
                f"Gemini API in backoff due to quota exhaustion. "
                f"Remaining: {remaining_sec:.1f}s. Using rule-based fallback."
            )
            raise QuotaExhaustedError(
                f"Gemini API quota exhausted. Backoff for {remaining_sec:.1f}s"
            )
        
        prompt = self._build_prompt(text, conversation_state, speaker_role=speaker_role)
        prompt_sent_wall_ms: Optional[int] = None
        if latency_context is not None:
            prompt_sent_wall_ms = log_gemini_prompt_sent(
                logger,
                latency_context,
                prompt_chars=len(prompt),
                speaker_role=speaker_role,
                provider='gemini',
            )

        try:
            if self.client is not None:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._generation_config(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                response_text = response.text or ''
            else:
                response_text, transport = generate_with_transport_chain(
                    api_key=self._api_key,
                    model_name=self.model_name,
                    prompt=prompt,
                    preferred_mode=self._cached_transport,
                    slot_index=self._slot_index,
                )
                self._cached_transport = transport

            if not response_text:
                logger.error("Gemini returned an empty response text.")
                return self._default_response(conversation_state)

            # Parse and validate the JSON response
            raw_data = json.loads(response_text)
            validated = validate_llm_response(raw_data)

            # SUCCESS: Reset backoff counter
            if self._consecutive_429_errors > 0:
                logger.info(f"Gemini API recovered after {self._consecutive_429_errors} consecutive 429 errors")
                self._consecutive_429_errors = 0
                self._backoff_until_ms = 0

            logger.debug(
                f"LLM analysis validated: "
                f"feedback='{validated.direct_feedback[:50] if validated.direct_feedback else 'none'}...', "
                f"confidence={validated.confidence:.2f}, "
                f"type={validated.feedback_type or 'none'}"
            )

            if latency_context is not None and prompt_sent_wall_ms is not None:
                llm_round_trip_ms = max(0, int(time.time() * 1000) - prompt_sent_wall_ms)
                log_gemini_response_received(
                    logger,
                    latency_context,
                    prompt_sent_wall_ms=prompt_sent_wall_ms,
                    response_chars=len(response_text),
                    llm_round_trip_ms=llm_round_trip_ms,
                    has_feedback=bool(validated.direct_feedback),
                    confidence=validated.confidence,
                    provider='gemini',
                )

            return {
                'direct_feedback': validated.direct_feedback,
                'confidence': validated.confidence,
                'feedback_type': validated.feedback_type,
                'conversation_state': validated.estado.to_dict(),
                'playbook_template_key': validated.playbook_template_key,
                'playbook_variables': dict(validated.playbook_variables),
            }

        except Exception as e:
            error_message = str(e)

            if is_auth_error_message(error_message) or (
                'INVALID_ARGUMENT' in error_message
                and 'API key' in error_message.lower()
            ):
                logger.error(
                    'Gemini auth failed for slot=%s prefix=%s — pool failover may retry another key',
                    self._slot_index,
                    self._api_key_prefix,
                )
                raise InvalidGeminiApiKeyError(
                    f'Gemini authentication failed for slot {self._slot_index}',
                ) from e

            # Detect 429 quota exceeded errors
            if '429' in error_message or 'ResourceExhausted' in error_message or 'quota' in error_message.lower():
                self._consecutive_429_errors += 1
                
                # Extract retry delay from error message if available
                retry_delay_sec = 60  # Default 60s backoff
                if 'retry in' in error_message.lower():
                    try:
                        # Parse "retry in 14.87268893s"
                        match = re.search(r'retry in ([\d.]+)s', error_message.lower())
                        if match:
                            retry_delay_sec = float(match.group(1))
                    except Exception:
                        pass
                
                # Exponential backoff: base_delay * 2^(consecutive_errors - 1), max 5 minutes
                exponential_delay = min(
                    retry_delay_sec * (2 ** (self._consecutive_429_errors - 1)),
                    300  # Max 5 minutes
                )
                
                self._backoff_until_ms = now_ms + int(exponential_delay * 1000)
                
                logger.error(
                    f"Gemini API quota exceeded (429). "
                    f"Consecutive errors: {self._consecutive_429_errors}. "
                    f"Backoff for {exponential_delay:.1f}s until {time.strftime('%H:%M:%S', time.localtime(self._backoff_until_ms / 1000))}"
                )
                
                # Re-raise so the fallback chain can handle it
                raise QuotaExhaustedError(
                    f"Gemini API quota exceeded. Backoff for {exponential_delay:.1f}s"
                ) from e
            
            # Non-429 error: log and return default
            logger.exception(f"Error during Gemini analysis (non-quota): {e}")
            return self._default_response(conversation_state)

    def _generation_config(self, **kwargs: Any) -> Any:
        if self._generation_config_factory is None:
            return kwargs
        return self._generation_config_factory(**kwargs)

    def _default_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return a safe fallback if Gemini fails."""
        return {
            'direct_feedback': '',
            'confidence': 0.0,
            'feedback_type': None,
            'conversation_state': state,
            'playbook_template_key': None,
            'playbook_variables': {},
        }

    def _build_prompt(
        self,
        text: str,
        state: Dict[str, Any],
        speaker_role: str = 'client',
    ) -> str:
        """Construct the LLM Prompt with the current conversation state and the new transcript.
        
        Uses few-shot learning examples to improve response consistency and quality.
        """
        state_str = json.dumps(state, ensure_ascii=False, indent=2)
        normalized_role = 'host' if str(speaker_role).lower() == 'host' else 'client'
        role_label = 'vendedor/host' if normalized_role == 'host' else 'cliente'
        feedback_rule = (
            'Este trecho é do vendedor/host: atualize o estado, mas responda sempre com '
            '`feedback`: null, `confidence`: 0.0 e `feedback_type`: null.'
            if normalized_role == 'host'
            else 'Este trecho é do cliente: gere feedback somente quando o trecho do cliente justificar; use o estado apenas como contexto.'
        )
        return f"""Você é um copiloto de vendas de baixa latência. Analise o NOVO TRECHO considerando o ESTADO ATUAL DA CONVERSA e retorne APENAS um JSON válido.

PAPEL DO TRECHO: {role_label}. {feedback_rule}

OBJETIVO
Detectar sinais táticos e gerar feedback curto e acionável para o vendedor.

PRIORIDADE
Detecte principalmente:
- objection
- opportunity
- rapport
- closing
- clarification
- risk

Na maioria dos casos, foque apenas nesses sinais ou retorne "feedback": null.

Com fase_spin="neutro", NÃO mencione metodologias de venda nem fases no texto do feedback.

FORMATO (todos os campos da raiz são obrigatórios)

{
  "feedback": string|null,
  "confidence": number,
  "feedback_type": string|null,
  "playbook_template_key": string|null,
  "playbook_variables": object|null,
  "estado": object
}

ESTADO — regras obrigatórias

- O campo "estado" DEVE existir em toda resposta.
- Quando nenhum campo do estado mudar neste trecho, retorne exatamente:
  "estado": {}
- Quando houver mudanças, inclua em "estado" SOMENTE os campos que mudaram (o servidor fará merge com o estado atual).
- Campos possíveis em "estado":
  interesse, resistencia, objecoes_detectadas, engajamento, fase_spin,
  proxima_pergunta_spin, alerta_risco_spin, product, pain_points, objections, claims

VENDEDOR / HOST

Se o papel for vendedor/host:
- NÃO gere feedback.
- Apenas atualize contexto em "estado" (product, pain_points, objections, claims, etc.).
- Responda sempre com:
  "feedback": null
  "confidence": 0.0
  "feedback_type": null
  "playbook_template_key": null
  "playbook_variables": null

PLAYBOOK

- Quando um playbook do tenant claramente se aplicar ao trecho:
  preencha "playbook_template_key" e "playbook_variables".
- Quando NÃO houver playbook aplicável, use explicitamente:
  "playbook_template_key": null
  "playbook_variables": null

SEM INTERVENÇÃO

Se não houver necessidade de intervir, use:
  "feedback": null
  "feedback_type": null

EXEMPLOS (formato resumido — não repita o estado inteiro se nada mudou)

Cliente: "Achei caro comparado ao concorrente X"
=> feedback_type="objection", objecoes_detectadas=["preco","concorrente"]

Cliente: "Me interessa. Qual o próximo passo?"
=> feedback_type="closing"

Cliente: "Pode continuar explicando."
=> "feedback": null, "estado": {}

Cliente: "Preciso pensar e decidir depois."
=> feedback_type="objection", objecoes_detectadas=["tempo"]

REGRAS DE SPIN (só com evidência clara)

Fases: situacao → problema → implicacao → necessidade
Mantenha a fase atual salvo nova evidência.

Somente quando fase_spin != "neutro":
- pode sugerir proxima_pergunta_spin
- pode mencionar descoberta/impacto no feedback

alerta_risco_spin=true apenas quando o vendedor avançar para proposta/fechamento
antes de entender adequadamente dor, impacto ou necessidade.
Nesse caso: feedback_type="risk"

CATEGORIAS DE OBJEÇÃO (em objecoes_detectadas)

preco, concorrente, tempo, confianca, funcionalidade, contrato, implementacao, roi

Em "objections", use a frase literal do cliente quando relevante.

REGRAS FINAIS

1. Analise o trecho considerando o estado atual.
2. Feedback: máximo 2 frases, específico e acionável.
3. Se não houver intervenção relevante, use "feedback": null.
4. Confiança:
   - 0.9–1.0: sinal explícito
   - 0.7–0.9: sinal claro
   - 0.5–0.7: ambíguo
   - abaixo disso: prefira "feedback": null
5. Retorne apenas JSON válido.

ESTADO ATUAL:
{state_str}

NOVO TRECHO ({role_label}):
{text}
"""
