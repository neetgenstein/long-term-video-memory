"""LLM provider support: OpenAI, Ollama, Gemini."""
import os

class LLMProvider:
    def generate(self, prompt):
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self, model="gpt-4o-mini", api_key=None):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model
            print(f"✓ Loaded OpenAI provider: {model}")
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI: {e}")
    
    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()

class OllamaProvider(LLMProvider):
    def __init__(self, model="llama3.2", host="http://localhost:11434"):
        try:
            import requests
            self.model = model
            self.host = host
            self.session = requests.Session()
            # Test connection
            test_url = f"{host}/api/tags"
            resp = self.session.get(test_url, timeout=5)
            resp.raise_for_status()
            print(f"✓ Loaded Ollama provider: {model} at {host}")
        except ImportError:
            raise ImportError("requests package not installed. Run: pip install requests")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama at {host}: {e}")
    
    def generate(self, prompt):
        import requests
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 150}
        }
        resp = self.session.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["response"].strip()

class GeminiProvider(LLMProvider):
    def __init__(self, model="gemini-2.0-flash-exp", api_key=None):
        try:
            import google.generativeai as genai
            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel(model)
            self.model_name = model
            print(f"✓ Loaded Gemini provider: {model}")
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini: {e}")
    
    def generate(self, prompt):
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 150}
        )
        return response.text.strip()

class MockProvider(LLMProvider):
    """Mock LLM for testing."""
    def __init__(self):
        print("✓ Using Mock LLM provider")
    
    def generate(self, prompt):
        return "Mock answer: This is a test response."
