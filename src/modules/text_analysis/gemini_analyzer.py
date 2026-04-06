"""Gemini-based semantic analyzer and state generator."""

import json
import logging
from typing import Any, Dict

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from .llm_state_validator import (
    validate_llm_response,
    LLMAnalysisResult,
    ConversationState,
)

logger = logging.getLogger(__name__)

class GeminiAnalyzer:
    """Analyze transcription texts and manage conversation state using Gemini Flash."""

    def __init__(self, api_key: str, model_name: str = 'gemini-2.0-flash'):
        if api_key:
            genai.configure(api_key=api_key)
        else:
            logger.warning("No Gemini API key provided. Analysis might fail if not injected properly.")
        
        self.model = genai.GenerativeModel(model_name)

    def analyze(self, text: str, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send the transcribed text and current conversational state to Gemini.
        Returns a dict containing 'direct_feedback' (str), 'confidence' (float),
        'feedback_type' (str or None), and 'conversation_state' (dict).
        """
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

        except json.JSONDecodeError as e:
            logger.error(f"Gemini returned invalid JSON: {e}")
            return self._default_response(conversation_state)
        except Exception as e:
            logger.exception(f"Error during Gemini analysis: {e}")
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
        return f"""Você é um assistente de vendas experiente agindo como um "co-piloto" para um representante comercial durante uma videochamada.

OBJETIVO: Analisar trechos de conversas de vendas em tempo real e fornecer feedback tático conciso para o vendedor.

# EXEMPLOS DE REFERÊNCIA

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
    "engajamento": "medio"
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
    "engajamento": "alto"
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
    "engajamento": "medio"
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
    "engajamento": "baixo"
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

# INSTRUÇÕES:
1. Analise o "Novo Trecho" considerando o "Estado Atual da Conversa"
2. Formule feedback tático curto (1-2 frases no máximo) SOMENTE quando necessário
3. Seja específico e acionável - evite feedback genérico
4. Se não houver necessidade de intervir, use feedback: null
5. Avalie sua confiança na análise (0.0 a 1.0):
   - 0.9-1.0: Sinal muito claro e explícito
   - 0.7-0.9: Sinal claro com bom contexto
   - 0.5-0.7: Sinal moderado, alguma ambiguidade
   - 0.0-0.5: Incerto, melhor não intervir
6. Atualize o estado da conversa mantendo a estrutura

ESTADO ATUAL DA CONVERSA:
{state_str}

NOVO TRECHO:
"{text}"

RESPONDA APENAS UM JSON VÁLIDO SEGUINDO O FORMATO DOS EXEMPLOS ACIMA.
"""
