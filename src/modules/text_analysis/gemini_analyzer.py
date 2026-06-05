"""Gemini-based semantic analyzer and state generator."""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

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


def create_genai_client(api_key: str, *, slot_index: Optional[int] = None) -> Any:
    """Create a google-genai client for legacy ``AIza…`` Developer API keys.

    ``AQ.…`` keys are Vertex AI Express credentials. They must call
    ``aiplatform.googleapis.com/v1/publishers/google/models/...?key=`` via
    :meth:`GeminiAnalyzer._rest_generate_content`, not ``generativelanguage``.
    """
    normalized = (api_key or '').strip()
    if uses_vertex_express_api(normalized):
        logger.info(
            'Gemini transport | mode=vertex_express_rest | slot=%s | key_prefix=%s',
            slot_index,
            normalized[:8] + '...',
        )
        return None

    from google import genai

    return genai.Client(api_key=normalized, vertexai=False)


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
        if client is None:
            if not api_key:
                logger.warning("No Gemini API key provided. Analysis might fail if not injected properly.")
            try:
                from google import genai
                from google.genai import types
            except Exception as exc:
                raise RuntimeError(
                    'google-genai is required for Gemini analysis. '
                    'Install python-service requirements before starting.',
                ) from exc

            self._api_key = (api_key or '').strip()
            self._use_rest_transport = uses_vertex_express_api(self._api_key)
            client = create_genai_client(api_key, slot_index=slot_index)
            self._generation_config_factory = (
                None if self._use_rest_transport else types.GenerateContentConfig
            )
        else:
            self._api_key = (api_key or '').strip()
            self._use_rest_transport = uses_vertex_express_api(self._api_key)
            self._generation_config_factory = None

        self.client = client
        self.model_name = model_name
        self._slot_index = slot_index
        self._api_key_prefix = (api_key[:8] + '...') if len(api_key) > 8 else 'unset'

        # Quota protection: track consecutive 429 errors
        self._consecutive_429_errors = 0
        self._backoff_until_ms = 0  # Don't call API until this timestamp

    def analyze(
        self,
        text: str,
        conversation_state: Dict[str, Any],
        speaker_role: str = 'client',
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

        try:
            if self._use_rest_transport:
                response_text = self._rest_generate_content(prompt)
            else:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._generation_config(
                        response_mime_type="application/json",
                        temperature=0.2,
                    )
                )
                response_text = response.text or ''

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

            if (
                'API_KEY_INVALID' in error_message
                or 'API key not valid' in error_message
                or (
                    'INVALID_ARGUMENT' in error_message
                    and 'API key' in error_message.lower()
                )
            ):
                logger.error(
                    'Gemini rejected API key | slot=%s | key_prefix=%s | '
                    'If you have multiple keys, use GEMINI_API_KEYS=key1,key2 '
                    '(not a comma-separated GEMINI_API_KEY).',
                    self._slot_index,
                    self._api_key_prefix,
                )
                raise InvalidGeminiApiKeyError(
                    'Gemini API key is invalid for this slot',
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

    def _rest_generate_content(self, prompt: str) -> str:
        """Call Vertex AI Express REST API (``AQ.…`` keys)."""
        import requests

        url = (
            'https://aiplatform.googleapis.com/v1/publishers/google/models/'
            f'{self.model_name}:generateContent'
        )
        response = requests.post(
            url,
            params={'key': self._api_key},
            headers={'Content-Type': 'application/json'},
            json={
                'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                'generationConfig': {
                    'responseMimeType': 'application/json',
                    'temperature': 0.2,
                },
            },
            timeout=60,
        )
        if not response.ok:
            logger.error(
                'Vertex express generateContent failed | slot=%s | status=%s | body=%s',
                self._slot_index,
                response.status_code,
                response.text[:500],
            )
            response.raise_for_status()
        payload = response.json()
        candidates = payload.get('candidates') or []
        if not candidates:
            return ''
        parts = (candidates[0].get('content') or {}).get('parts') or []
        texts = [part.get('text', '') for part in parts if isinstance(part, dict)]
        return ''.join(texts).strip()
            
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
        return f"""Você é um motor de IA de baixa latência agindo como um "co-piloto" tático para um representante comercial durante uma videochamada. Sua função é analisar trechos de conversas e fornecer feedbacks concisos e acionáveis.

OBJETIVO: Analisar conversas de vendas em tempo real e fornecer feedback tático conciso para o vendedor.

PRIORIDADE: Na maior parte dos trechos, concentre-se apenas em sinais táticos: objeção, oportunidade, rapport, fechamento, ou nenhuma intervenção (`feedback`: null). O texto do `feedback` não deve nomear metodologias de venda nem rotular "fases" de descoberta, **salvo** quando `fase_spin` no estado já não for `neutro` ou quando `alerta_risco_spin` for aplicável (campos opcionais descritos mais abaixo).

PAPEL DO TRECHO ATUAL: {role_label}. {feedback_rule}

# EXEMPLOS DE REFERÊNCIA (caminho principal — maioria dos trechos)

Exemplo 1 - Objeção de preço:
Trecho: "Achei caro comparado ao concorrente X"
Resposta esperada:
{{
  "feedback": "Cliente comparou preço - destaque diferenciais e ROI vs concorrente X",
  "confidence": 0.9,
  "feedback_type": "objection",
  "estado": {{
    "interesse": "medio",
    "resistencia": "alta",
    "objecoes_detectadas": ["preco", "concorrente"],
    "engajamento": "medio",
    "fase_spin": "neutro",
    "proxima_pergunta_spin": "",
    "alerta_risco_spin": false,
    "product": "",
    "pain_points": [],
    "objections": ["Achei caro comparado ao concorrente X"],
    "claims": []
  }}
}}

Exemplo 2 - Sinal de compra:
Trecho: "Ok, me interessa. Como funciona o próximo passo?"
Resposta esperada:
{{
  "feedback": "Sinal de compra detectado! Apresente próximo passo claro (proposta, contrato, implementação)",
  "confidence": 0.95,
  "feedback_type": "closing",
  "estado": {{
    "interesse": "alto",
    "resistencia": "baixa",
    "objecoes_detectadas": [],
    "engajamento": "alto",
    "fase_spin": "neutro",
    "proxima_pergunta_spin": "",
    "alerta_risco_spin": false,
    "product": "",
    "pain_points": [],
    "objections": [],
    "claims": []
  }}
}}

Exemplo 3 - Sem intervenção necessária:
Trecho: "Ok, entendi. Pode continuar explicando"
Resposta esperada:
{{
  "feedback": null,
  "confidence": 0.8,
  "feedback_type": null,
  "estado": {{
    "interesse": "medio",
    "resistencia": "baixa",
    "objecoes_detectadas": [],
    "engajamento": "medio",
    "fase_spin": "neutro",
    "proxima_pergunta_spin": "",
    "alerta_risco_spin": false,
    "product": "",
    "pain_points": [],
    "objections": [],
    "claims": []
  }}
}}

Exemplo 4 - Objeção de tempo:
Trecho: "Preciso pensar, me liga mês que vem"
Resposta esperada:
{{
  "feedback": "Objeção de tempo - crie urgência com benefícios de começar agora",
  "confidence": 0.85,
  "feedback_type": "objection",
  "estado": {{
    "interesse": "medio",
    "resistencia": "alta",
    "objecoes_detectadas": ["tempo"],
    "engajamento": "baixo",
    "fase_spin": "neutro",
    "proxima_pergunta_spin": "",
    "alerta_risco_spin": false,
    "product": "",
    "pain_points": [],
    "objections": ["Preciso pensar, me liga mês que vem"],
    "claims": []
  }}
}}

# Campos opcionais no estado (referência SPIN — uso secundário)

Só preencha estes campos quando o trecho (e o estado atual) derem suporte; na dúvida mantenha `fase_spin`: **"neutro"** e `proxima_pergunta_spin` vazio. Referência rápida das fases: **situacao** (contexto) → **problema** (dor) → **implicacao** (impacto se não resolver) → **necessidade** (valor da solução). Preserve `fase_spin` entre trechos salvo evidência nova clara. `alerta_risco_spin`: true só se o vendedor **pular** etapas (ex.: proposta antes de dor/impacto claros) — aí use `feedback_type`: **"risk"**. `proxima_pergunta_spin`: só se `fase_spin` não for neutro e fizer sentido.

# Exemplos adicionais (só quando o trecho justificar — não é o padrão)

Exemplo 5 - Dor explícita do cliente (fase problema + pergunta sugerida):
Trecho do cliente: "Hoje a equipe perde um dia inteiro fechando a folha manualmente."
Resposta esperada:
{{
  "feedback": "Cliente descreveu um gargalo operacional — explore impacto (tempo/custo) antes de apresentar solução",
  "confidence": 0.82,
  "feedback_type": "opportunity",
  "estado": {{
    "interesse": "medio",
    "resistencia": "baixa",
    "objecoes_detectadas": [],
    "engajamento": "medio",
    "fase_spin": "problema",
    "proxima_pergunta_spin": "Quanto isso custa em horas ou reais por mês para vocês?",
    "alerta_risco_spin": false,
    "product": "",
    "pain_points": ["Equipe perde um dia inteiro fechando a folha manualmente"],
    "objections": [],
    "claims": []
  }}
}}

Exemplo 6 - Risco: solução cedo demais (`feedback_type` risk):
Estado atual já tinha: "fase_spin": "problema"
Trecho do vendedor: "Posso te mandar a proposta fechada ainda hoje com implantação na próxima semana."
Resposta esperada:
{{
  "feedback": "Risco: fechamento antes de impacto/valor claros — confirme necessidade e custo do problema antes da proposta",
  "confidence": 0.78,
  "feedback_type": "risk",
  "estado": {{
    "interesse": "medio",
    "resistencia": "baixa",
    "objecoes_detectadas": [],
    "engajamento": "medio",
    "fase_spin": "problema",
    "proxima_pergunta_spin": "Se nada mudar, qual o custo disso nos próximos 6 meses?",
    "alerta_risco_spin": true,
    "product": "",
    "pain_points": [],
    "objections": [],
    "claims": ["Proposta fechada com implantação na próxima semana"]
  }}
}}

Exemplo 7 - Com playbook (opcional):
Trecho: "O produto do concorrente X está mais barato que o vosso."
Resposta esperada (note `playbook_template_key` e `playbook_variables` na raiz):
{{
  "feedback": "Cliente comparou com concorrente — contraste valor/diferenciação antes de discutir só preço",
  "confidence": 0.88,
  "feedback_type": "objection",
  "playbook_template_key": "preco_vs_concorrente",
  "playbook_variables": {{
    "competidor": "X"
  }},
  "estado": {{
    "interesse": "medio",
    "resistencia": "alta",
    "objecoes_detectadas": ["preco", "concorrente"],
    "engajamento": "medio",
    "fase_spin": "neutro",
    "proxima_pergunta_spin": "",
    "alerta_risco_spin": false,
    "product": "",
    "pain_points": [],
    "objections": ["Produto do concorrente X está mais barato"],
    "claims": []
  }}
}}

# CATEGORIAS VÁLIDAS PARA objecoes_detectadas:
- preco: preocupações com preço/custo
- concorrente: comparações com concorrentes
- tempo: objeções de timing ("preciso pensar", "me liga depois")
- confianca: dúvidas de confiança/credibilidade
- funcionalidade: limitações de funcionalidades
- contrato: preocupações com termos contratuais
- implementacao: preocupações com implementação
- roi: dúvidas sobre retorno do investimento

# TIPOS VÁLIDOS PARA feedback_type:
- objection: objeção detectada
- opportunity: oportunidade identificada
- rapport: momento de conexão pessoal
- closing: sinal de compra/fechamento
- clarification: precisa esclarecer algo
- risk: risco potencial na negociação

# VALORES VÁLIDOS PARA fase_spin:
- neutro | situacao | problema | implicacao | necessidade

# CAMPOS DE RESUMO INCREMENTAL DA REUNIÃO:
- `product`: produto/solução discutida; mantenha o valor anterior salvo quando o trecho trouxer um nome mais específico.
- `pain_points`: dores explícitas do cliente, em frases curtas.
- `objections`: objeções em linguagem livre, preservando contexto concreto.
- `claims`: promessas, benefícios ou afirmações importantes ditas na call.
- Adicione itens novos às listas; não apague itens já presentes no Estado Atual salvo correção explícita. O serviço fará dedup/cap.

# INSTRUÇÕES:
1. Primeiro identifique sinais táticos como nos exemplos 1–4; só depois avalie se os campos opcionais de fase (seção SPIN acima) se aplicam.
2. Analise o "Novo Trecho" considerando o "Estado Atual da Conversa" (incluindo `fase_spin` anterior).
3. Formule feedback tático curto (1-2 frases no máximo) SOMENTE quando necessário; com `fase_spin` neutro, **não** mencione metodologia nem rotule fases no texto.
4. Seja específico e acionável - evite feedback genérico
5. Se não houver necessidade de intervir, use feedback: null
6. Avalie sua confiança na análise (0.0 a 1.0):
   - 0.9-1.0: Sinal muito claro e explícito
   - 0.7-0.9: Sinal claro com bom contexto
   - 0.5-0.7: Sinal moderado, alguma ambiguidade
   - 0.0-0.5: Incerto, melhor não intervir
7. Atualize o estado da conversa mantendo **todos** os campos do exemplo (interesse, resistencia, objecoes_detectadas, engajamento, fase_spin, proxima_pergunta_spin, alerta_risco_spin, product, pain_points, objections, claims).
8. Transição de `fase_spin`: só com evidência; caso contrário mantenha o valor já presente no estado atual.
9. Se o papel do trecho atual for vendedor/host, use o trecho apenas para enriquecer `product` e `claims`/contexto; não gere feedback.

# PLAYBOOK (opcional — raiz do JSON, fora de `estado`)

Quando um **roteiro acionável** cadastrado no tenant claramente se aplica ao trecho (ex.: objeção de preço → template `preco`), inclua no objeto raiz:
- `playbook_template_key`: string curta (slug do template, máx. 64 caracteres). Omita ou use `null` se não houver template aplicável.
- `playbook_variables`: objeto com strings para interpolar placeholders nos passos do template (ex.: `{{"competidor": "Concorrente X", "produto": "Suite Pro"}}`). No máximo ~32 chaves; valores curtos.

Se não tiver certeza, omita ambos ou use `null`.

ESTADO ATUAL DA CONVERSA:
{state_str}

NOVO TRECHO ({role_label}):
"{text}"

RESPONDA APENAS UM JSON VÁLIDO SEGUINDO O FORMATO DOS EXEMPLOS ACIMA.
"""
