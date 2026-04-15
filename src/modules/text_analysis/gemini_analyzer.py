"""Gemini-based semantic analyzer and state generator."""

import json
import logging
import time
from typing import Any, Dict

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from .llm_state_validator import (
    validate_llm_response,
    LLMAnalysisResult,
    ConversationState,
)

logger = logging.getLogger(__name__)


class QuotaExhaustedError(Exception):
    """Raised when Gemini API quota is exhausted and we need to back off.
    
    This allows the fallback chain to distinguish between:
    - Temporary errors (network, timeout) → retry
    - Quota exhaustion (429) → use rule-based fallback immediately
    """
    pass


class GeminiAnalyzer:
    """Analyze transcription texts and manage conversation state using Gemini Flash.
    
    Features:
    - Quota protection: backs off on 429 errors to avoid hammering the API
    - Graceful degradation: returns empty fallback instead of crashing
    - Response validation: ensures LLM output matches expected schema
    """

    def __init__(self, api_key: str, model_name: str = 'gemini-2.5-flash'):
        if api_key:
            genai.configure(api_key=api_key)
        else:
            logger.warning("No Gemini API key provided. Analysis might fail if not injected properly.")

        self.model = genai.GenerativeModel(model_name)
        
        # Quota protection: track consecutive 429 errors
        self._consecutive_429_errors = 0
        self._backoff_until_ms = 0  # Don't call API until this timestamp

    def analyze(self, text: str, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
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
        
        prompt = self._build_prompt(text, conversation_state)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                )
            )

            if not response.text:
                logger.error("Gemini returned an empty response text.")
                return self._default_response(conversation_state)

            # Parse and validate the JSON response
            raw_data = json.loads(response.text)
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
                'conversation_state': validated.estado.to_dict()
            }

        except Exception as e:
            error_message = str(e)
            
            # Detect 429 quota exceeded errors
            if '429' in error_message or 'ResourceExhausted' in error_message or 'quota' in error_message.lower():
                self._consecutive_429_errors += 1
                
                # Extract retry delay from error message if available
                retry_delay_sec = 60  # Default 60s backoff
                if 'retry in' in error_message.lower():
                    try:
                        # Parse "retry in 14.87268893s"
                        import re
                        match = re.search(r'retry in ([\d.]+)s', error_message.lower())
                        if match:
                            retry_delay_sec = float(match.group(1))
                    except:
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
            
    def _default_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return a safe fallback if Gemini fails."""
        return {
            'direct_feedback': '',
            'confidence': 0.0,
            'feedback_type': None,
            'conversation_state': state
        }

    def _build_prompt(self, text: str, state: Dict[str, Any]) -> str:
        """Construct the LLM Prompt with the current conversation state and the new transcript.
        
        Uses few-shot learning examples to improve response consistency and quality.
        """
        state_str = json.dumps(state, ensure_ascii=False, indent=2)
        return f"""Você é um motor de IA de baixa latência agindo como um "co-piloto" tático para um representante comercial durante uma videochamada. Sua função é analisar trechos de conversas e fornecer feedbacks concisos e acionáveis.

OBJETIVO: Analisar conversas de vendas em tempo real e fornecer feedback tático conciso para o vendedor.

PRIORIDADE: Na maior parte dos trechos, concentre-se apenas em sinais táticos: objeção, oportunidade, rapport, fechamento, ou nenhuma intervenção (`feedback`: null). O texto do `feedback` não deve nomear metodologias de venda nem rotular "fases" de descoberta, **salvo** quando `fase_spin` no estado já não for `neutro` ou quando `alerta_risco_spin` for aplicável (campos opcionais descritos mais abaixo).

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
    "alerta_risco_spin": false
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
    "alerta_risco_spin": false
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
    "alerta_risco_spin": false
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
    "alerta_risco_spin": false
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
    "alerta_risco_spin": false
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
    "alerta_risco_spin": true
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
7. Atualize o estado da conversa mantendo **todos** os campos do exemplo (interesse, resistencia, objecoes_detectadas, engajamento, fase_spin, proxima_pergunta_spin, alerta_risco_spin).
8. Transição de `fase_spin`: só com evidência; caso contrário mantenha o valor já presente no estado atual.

ESTADO ATUAL DA CONVERSA:
{state_str}

NOVO TRECHO:
"{text}"

RESPONDA APENAS UM JSON VÁLIDO SEGUINDO O FORMATO DOS EXEMPLOS ACIMA.
"""
