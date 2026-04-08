"""Ollama-based LLM analyzer for 100% free local inference.

Ollama runs open-source models locally (no API costs, no quotas).
Supports models like:
- llama3.1:8b (fast, good for sales analysis)
- llama3.1:70b (slower but better quality)
- mistral:7b (good balance)
- qwen2.5:7b (multilingual, excellent for PT-BR)

Setup:
1. Install Ollama: https://ollama.ai
2. Pull model: ollama pull llama3.1:8b
3. Configure: OLLAMA_BASE_URL=http://localhost:11434
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Optional

import requests

from .llm_state_validator import (
    validate_llm_response,
    ConversationState,
)

logger = logging.getLogger(__name__)


class OllamaAnalyzer:
    """Analyze sales conversations using locally-hosted Ollama models.
    
    Benefits:
    - 100% FREE (no API costs, no quotas)
    - No rate limits (runs on your hardware)
    - Data privacy (everything stays local)
    - Works offline after model download
    
    Performance:
    - llama3.1:8b: ~50-100ms/token on modern CPU
    - qwen2.5:7b: ~40-80ms/token (better for PT-BR)
    - llama3.1:70b: ~200-400ms/token (best quality)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: int = 30,
    ):
        """
        Args:
            base_url: Ollama API endpoint (default: http://localhost:11434)
            model: Model name to use (e.g., 'llama3.1:8b', 'qwen2.5:7b')
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        
        # Test connection
        try:
            self._check_connection()
            logger.info(f"Ollama connected at {base_url} using model '{model}'")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {base_url}: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            raise

    def _check_connection(self) -> None:
        """Test connection to Ollama API and verify model is available."""
        # Check if Ollama is running
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        
        available_models = [m["name"] for m in response.json().get("models", [])]
        
        if self.model not in available_models:
            logger.warning(
                f"Model '{self.model}' not found in Ollama. "
                f"Available models: {available_models}"
            )
            logger.warning(f"Run: ollama pull {self.model}")
        else:
            logger.info(f"Model '{self.model}' is available in Ollama")

    def analyze(self, text: str, conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze transcribed text using Ollama local model.
        Returns same format as GeminiAnalyzer for drop-in replacement.
        """
        prompt = self._build_prompt(text, conversation_state)

        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 200,  # Reduced from 500 - we only need JSON
                        "num_ctx": 2048,     # Reduced from default 4096 - faster on CPU
                    }
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            duration_ms = (time.time() - start_time) * 1000
            result = response.json()
            
            response_text = result.get("response", "").strip()
            
            if not response_text:
                logger.error("Ollama returned empty response")
                return self._default_response(conversation_state)
            
            # Extract JSON from response (Ollama might add extra text)
            json_str = self._extract_json(response_text)
            
            if not json_str:
                logger.error(f"Could not extract JSON from Ollama response: {response_text[:200]}")
                return self._default_response(conversation_state)
            
            # Parse and validate
            raw_data = json.loads(json_str)
            validated = validate_llm_response(raw_data)
            
            logger.debug(
                f"Ollama analysis ({duration_ms:.0f}ms): "
                f"feedback='{validated.direct_feedback[:50] if validated.direct_feedback else 'none'}...', "
                f"confidence={validated.confidence:.2f}"
            )
            
            return {
                'direct_feedback': validated.direct_feedback,
                'confidence': validated.confidence,
                'feedback_type': validated.feedback_type,
                'conversation_state': validated.estado.to_dict()
            }

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to Ollama at {self.base_url}: {e}")
            raise OllamaConnectionError(
                f"Ollama is not running at {self.base_url}. Start it with: ollama serve"
            ) from e
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            raise OllamaTimeoutError(
                f"Ollama request timed out. Model might be busy or too large for your hardware."
            )
        except Exception as e:
            logger.exception(f"Error during Ollama analysis: {e}")
            return self._default_response(conversation_state)

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from Ollama response.
        
        Ollama might add conversational text before/after the JSON.
        We look for the first complete JSON object.
        """
        # Try to find JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        
        return None

    def _default_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return a safe fallback if Ollama fails."""
        return {
            'direct_feedback': '',
            'confidence': 0.0,
            'feedback_type': None,
            'conversation_state': state
        }

    def _build_prompt(self, text: str, state: Dict[str, Any]) -> str:
        """Build optimized prompt for local Ollama models.
        
        Returns precise, actionable feedback in Portuguese for sales calls.
        """
        state_str = json.dumps(state, ensure_ascii=False)
        return f"""Você é um assistente de vendas experiente analisando uma conversa em tempo real.

Analise o trecho e dê feedback TÁTICO e ESPECÍFICO em PORTUGUÊS para o vendedor.

ESTADO ATUAL:
{state_str}

TRECHO DA CONVERSA:
"{text}"

Regras:
- Responda SOMENTE em português
- Feedback deve ser ação concreta e específica (não genérica)
- Máximo 1 frase curta
- Se não houver sinal claro de objeção ou oportunidade, responda null
- Confiança: 0.9-1.0 (sinal explícito), 0.7-0.9 (sinal claro), 0.5-0.7 (moderado), <0.5 (incerto)

Responda APENAS JSON válido:
{{"feedback":"conselho tático em português ou null","confidence":0.85,"feedback_type":"objection|opportunity|rapport|closing|clarification|risk|null","estado":{{"interesse":"baixo|medio|alto","resistencia":"baixa|media|alta","objecoes_detectadas":["preco","concorrente","tempo","confianca","funcionalidade","contrato","implementacao","roi"],"engajamento":"baixo|medio|alto"}}}}
"""


class OllamaConnectionError(Exception):
    """Raised when cannot connect to Ollama service."""
    pass


class OllamaTimeoutError(Exception):
    """Raised when Ollama request times out."""
    pass
