import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

class LLMClient:
    def __init__(self):
        self.mock = os.getenv("LLM_MOCK", "true").lower() == "true"
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        # OpenAI setup
        self.openai_client = None
        if self.provider == "openai" and not self.mock:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY missing in .env")
            self.openai_client = OpenAI(api_key=api_key)

        # Ollama setup
        self.ollama_url = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama2")

    def ask(self, prompt: str, stream: bool = False):
        """
        Ask the LLM for a response.
        :param prompt: User query
        :param stream: Whether to stream the response
        :return: Dict with provider, model, response (string or generator if stream=True)
        """
        if self.mock:
            return {"provider": self.provider, "model": self.model, "response": f"Mock reply to: {prompt}"}

        if self.provider == "openai":
            try:
                if stream:
                    # Stream response
                    stream_resp = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True
                    )
                    def generator():
                        for chunk in stream_resp:
                            if chunk.choices and chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content
                    return {"provider": "openai", "model": self.model, "response": generator()}
                else:
                    resp = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return {"provider": "openai", "model": self.model, "response": resp.choices[0].message.content}
            except Exception as e:
                return {"error": str(e)}

        elif self.provider == "ollama":
            try:
                payload = {"model": self.ollama_model, "prompt": prompt, "stream": stream}
                resp = requests.post(self.ollama_url, json=payload, stream=stream)
                resp.raise_for_status()

                if stream:
                    def generator():
                        for line in resp.iter_lines():
                            if line:
                                try:
                                    data = line.decode("utf-8")
                                    yield data
                                except Exception:
                                    continue
                    return {"provider": "ollama", "model": self.ollama_model, "response": generator()}
                else:
                    return {
                        "provider": "ollama",
                        "model": self.ollama_model,
                        "response": resp.json().get("response", "")
                    }
            except Exception as e:
                return {"error": str(e)}

        else:
            return {"error": f"Unsupported LLM provider: {self.provider}"}


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    llm = LLMClient()

    # Normal response
    reply = llm.ask("Summarize the benefits of self-healing systems.")
    print("LLM Response:", reply)

    # Streaming response
    stream_reply = llm.ask("Explain how self-healing AI assistants work.", stream=True)
    print("\nStreaming Response:")
    for chunk in stream_reply["response"]:
        print(chunk, end="", flush=True)