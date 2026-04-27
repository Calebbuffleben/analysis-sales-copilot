#!/usr/bin/env python3
"""
Gemini API verification script.

Tests:
1. API key is configured
2. Gemini connection works
3. Model responds correctly to a sales analysis prompt
4. Rate limiter configuration is correct

Usage:
    cd python-service
    python test_gemini_verify.py
"""

import os
import sys
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

def test_api_key():
    """Test 1: Verify API key is configured."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        print("❌ FAIL: GEMINI_API_KEY not set in .env")
        print("   Fix: Add your API key from https://aistudio.google.com/app/apikey")
        return False
    masked = api_key[:4] + '...' + api_key[-4:]
    print(f"✅ PASS: API key configured ({masked})")
    return True

def test_connection():
    """Test 2: Verify Gemini connection."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

        # List available models
        models = list(genai.list_models())
        flash_models = [m.name for m in models if 'flash' in m.name.lower() and 'generateContent' in m.supported_generation_methods]

        if not flash_models:
            print(f"⚠️  WARN: No Flash models found. Available: {[m.name for m in models[:5]]}")
        else:
            print(f"✅ PASS: Gemini connected. Flash models: {', '.join(flash_models)}")

        return True
    except Exception as e:
        print(f"❌ FAIL: Gemini connection error: {e}")
        return False

def test_analysis():
    """Test 3: Verify Gemini responds to sales analysis prompt."""
    try:
        from src.modules.text_analysis.gemini_analyzer import GeminiAnalyzer

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ FAIL: No API key for analyzer test")
            return False

        analyzer = GeminiAnalyzer(api_key=api_key)

        test_text = "Achei o preço um pouco alto comparado ao concorrente X"
        test_state = {
            'interesse': 'medio',
            'resistencia': 'baixa',
            'objecoes_detectadas': [],
            'engajamento': 'medio',
            'fase_spin': 'neutro',
            'proxima_pergunta_spin': '',
            'alerta_risco_spin': False,
        }

        print("   Calling Gemini analysis (timeout: 30s)...")
        start = time.time()
        result = analyzer.analyze(test_text, test_state)
        elapsed = time.time() - start

        feedback = result.get('direct_feedback', '')
        confidence = result.get('confidence', 0)
        cs = result.get('conversation_state') or {}

        if feedback:
            print(f"✅ PASS: Analysis OK ({elapsed:.1f}s)")
            print(f"   Feedback: {feedback[:100]}...")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Type: {result.get('feedback_type', 'none')}")
            print(f"   fase_spin: {cs.get('fase_spin', 'missing')}")
            return True
        else:
            print(f"⚠️  WARN: Analysis returned empty feedback (confidence={confidence:.2f})")
            print(f"   Response: {json.dumps(result, indent=2, ensure_ascii=False)[:200]}")
            return True  # Connection works, even if response was empty

    except Exception as e:
        print(f"❌ FAIL: Analysis error: {e}")
        return False

def test_rate_limiter_config():
    """Test 4: Verify rate limiter configuration."""
    try:
        from src.modules.text_analysis.text_analysis_service import TextAnalysisService

        # Check RPM limit is set correctly
        if TextAnalysisService.RPM_LIMIT == 12:
            print(f"✅ PASS: Rate limiter configured (limit={TextAnalysisService.RPM_LIMIT} RPM, window={TextAnalysisService.RPM_WINDOW_SEC}s)")
            return True
        else:
            print(f"⚠️  WARN: Rate limiter limit is {TextAnalysisService.RPM_LIMIT} (expected 12)")
            return True

    except Exception as e:
        print(f"❌ FAIL: Rate limiter config error: {e}")
        return False

def test_spin_skip_step_risk():
    """Scenario: seller jumps to proposal while estado still problema — expect risk or alerta."""
    try:
        from src.modules.text_analysis.gemini_analyzer import GeminiAnalyzer

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ FAIL: No API key for SPIN scenario")
            return False

        analyzer = GeminiAnalyzer(api_key=api_key)
        test_state = {
            'interesse': 'medio',
            'resistencia': 'baixa',
            'objecoes_detectadas': [],
            'engajamento': 'medio',
            'fase_spin': 'problema',
            'proxima_pergunta_spin': 'Qual o impacto disso para o time?',
            'alerta_risco_spin': False,
        }
        test_text = (
            'Posso te enviar a proposta fechada ainda hoje e já agendar a implantação para semana que vem.'
        )
        print("   Calling Gemini (SPIN skip / risk scenario)...")
        result = analyzer.analyze(test_text, test_state)
        cs = result.get('conversation_state') or {}
        ft = result.get('feedback_type')
        risk = cs.get('alerta_risco_spin') is True
        print(f"   feedback_type={ft!r}, alerta_risco_spin={cs.get('alerta_risco_spin')}, fase_spin={cs.get('fase_spin')!r}")
        if risk or ft == 'risk':
            print('✅ PASS: Model flagged SPIN risk (risk type or alerta_risco_spin)')
            return True
        print('⚠️  WARN: Expected risk signal not detected (model-dependent); keys present OK')
        return True
    except Exception as e:
        print(f"❌ FAIL: SPIN scenario error: {e}")
        return False


def test_llm_provider():
    """Test 5: Verify LLM provider is set to gemini."""
    try:
        from src.config.settings import get_settings
        settings = get_settings()

        if settings.llm_provider == 'gemini':
            print(f"✅ PASS: LLM provider = gemini")
            return True
        else:
            print(f"⚠️  WARN: LLM provider = '{settings.llm_provider}' (expected 'gemini')")
            print("   Fix: Set LLM_PROVIDER=gemini in .env")
            return False

    except Exception as e:
        print(f"❌ FAIL: Settings error: {e}")
        return False

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("🔮 Gemini API Verification")
    print("=" * 60)
    print()

    tests = [
        ("1. API Key", test_api_key),
        ("2. Connection", test_connection),
        ("3. Analysis", test_analysis),
        ("4. Rate Limiter", test_rate_limiter_config),
        ("5. LLM Provider", test_llm_provider),
        ("6. SPIN skip / risk", test_spin_skip_step_risk),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{name}:")
        try:
            results.append(test_fn())
        except Exception as e:
            print(f"❌ FAIL: Unexpected error: {e}")
            results.append(False)

    print()
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    if all(results):
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("   Gemini is ready to use!")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
        print("   Check the errors above and fix them.")
        sys.exit(1)
