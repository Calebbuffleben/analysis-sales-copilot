"""Gemini-based semantic analyzer and state generator."""

import json
import logging
from typing import Any, Dict

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

logger = logging.getLogger(__name__)

class GeminiAnalyzer:
    """Analyze transcription texts and manage conversation state using Gemini Flash."""

    def __init__(self, api_key: str, model_name: str = 'gemini-1.5-flash'):
        if api_key:
            genai.configure(api_key=api_key)
        else:
            logger.warning("No Gemini API key provided. Analysis might fail if not injected properly.")
        
        self.model = genai.GenerativeModel(model_name)

    def analyze(self, text: str, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send the transcribed text and current conversational state to Gemini.
        Returns a dict containing 'direct_feedback' (str) and 'conversation_state' (dict).
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
                
            data = json.loads(response.text)
            direct_feedback = data.get('feedback') or ""
            new_state = data.get('estado', conversation_state)
            
            return {
                'direct_feedback': direct_feedback.strip() if isinstance(direct_feedback, str) else "",
                'conversation_state': new_state
            }
            
        except Exception as e:
            logger.exception("Error during Gemini analysis")
            return self._default_response(conversation_state)
            
    def _default_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return a safe fallback if Gemini fails."""
        return {
            'direct_feedback': '',
            'conversation_state': state
        }

    def _build_prompt(self, text: str, state: Dict[str, Any]) -> str:
        """Construct the LLM Prompt with the current conversation state and the new transcript."""
        state_str = json.dumps(state, ensure_ascii=False, indent=2)
        return f"""Você é um assistente de vendas experiente agindo como um "co-piloto" para um representante comercial durante uma videochamada.

Abaixo você receberá o "Estado Atual da Conversa" e um "Novo Trecho" de áudio transcrito (pode ser a fala do cliente ou do vendedor, ou um ruído sem sentido).

INSTRUÇÕES:
1. Analise o "Novo Trecho" levando em consideração o "Estado Atual da Conversa".
2. Formule um feedback curto e tático (ação recomendada) para mostrar na tela do vendedor, SOMENTE CASO SEJA NECESSÁRIO (ex: objeção detectada, momento de fechamento propício).
3. Seja breve! No máximo 1 ou 2 frases. Se não houver necessidade de intervir, o campo "feedback" deve ser nulo ou uma string vazia "".
4. Atualize o Estado da Conversa. Você deve manter preferencialmente as seguintes propriedades, ajustando os valores conforme a evolução da conversa:
  - interesse: "baixo", "medio", "alto"
  - resistencia: "baixa", "media", "alta"
  - objecoes_detectadas: lista de strings (ex: ["preco", "produto concorrente"])
  - engajamento: "baixo", "medio", "alto"
Você pode adicionar contextos a mais no objeto de estado se achar vital para as próximas predições.

ESTADO ATUAL DA CONVERSA:
{state_str}

NOVO TRECHO:
"{text}"

RESPONDA APENAS UM JSON VÁLIDO NO SEGUINTE FORMATO:
{{
  "feedback": "Sua sugestão aqui ou string vazia",
  "estado": {{
    "interesse": "medio",
    "resistencia": "media",
    "objecoes_detectadas": [],
    "engajamento": "medio"
  }}
}}
"""
