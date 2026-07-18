#!/usr/bin/env python3
"""
Lab 6: LLM Provider — Unified interface for cloud and local models
═══════════════════════════════════════════════════════════════════════
Provides a single get_llm() function that returns the right LLM backend:

  - If GROQ_API_KEY is set →  Groq            (cloud — free tier, fast)
  - Elif HF_TOKEN is set   →  HF Inference Providers (cloud — metered credits)
  - Otherwise              →  Ollama local model     (Codespaces / laptop)

All backends expose the same .invoke(messages) interface, so the rest
of the application code doesn't need to know which one is running.

Why Groq first?
---------------
Hugging Face replaced the old free serverless "Inference API" with metered
"Inference Providers". A free HF account now gets only ~$0.10/month of
credits, which this agent burns through in a couple of queries (the TAO loop
makes several LLM calls per question, each resending the full system prompt).
Once the credits are gone you get: 402 Client Error: Payment Required.

Groq's free tier is rate-limited (429, recovers) rather than credit-limited
(402, dead until you pay), so it's a better fit for a workshop. Each attendee
uses their own free Groq key. The HF path below still works if you have
PRO/pre-paid credits.
"""

import os

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 1.  Configuration                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
# Groq is OpenAI-compatible, so we reuse the openai SDK (already a
# dependency) instead of adding another package.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq retires models periodically (see console.groq.com/docs/deprecations).
# meta-llama/llama-4-scout was shut down 2026-07-17, which is why the old ID
# now returns 404 model_not_found.
#
# We use llama-3.3-70b-versatile: it's a plain instruct model that follows the
# hand-rolled text TAO protocol (Thought/Action/Args) this agent parses, so it's
# a true drop-in for scout. NOTE: Groq has this model slated for shutdown on
# 2026-08-16 — revisit before then.
#
# Why not openai/gpt-oss-* (Groq's recommended replacement)? Those are reasoning
# + native-tool-calling models. Live testing showed they DON'T drop into this
# agent's text protocol: default effort buries the answer in a separate
# `reasoning` field (empty .content), and reasoning_effort="low" makes the model
# emit a native tool call, which 400s here. Moving to gpt-oss would require
# prompt + wrapper rework and per-lab retesting — see notes for that migration.
GROQ_MODEL = "llama-3.3-70b-versatile"

HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

OLLAMA_MODEL = "llama3.2:latest"


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 2.  Response wrapper                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
class LLMResponse:
    """Simple wrapper so cloud responses look like LangChain responses."""
    def __init__(self, content: str):
        self.content = content


# Backwards-compatible alias (earlier labs referred to HFResponse)
HFResponse = LLMResponse


def _to_message_dicts(messages) -> list:
    """Normalise LangChain-style or dict-style messages to plain dicts."""
    out = []
    for msg in messages:
        if isinstance(msg, dict):
            out.append({"role": msg["role"], "content": msg["content"]})
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            out.append({"role": msg.role, "content": msg.content})
    return out


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 3.  Groq wrapper (free tier — preferred cloud backend)          ║
# ╚══════════════════════════════════════════════════════════════════╝
class GroqLLMWrapper:
    """
    Wraps the Groq API so it has the same .invoke(messages) interface
    as LangChain's ChatOllama.

    Groq exposes an OpenAI-compatible endpoint, so we point the openai
    SDK at Groq's base_url — no extra dependency needed.
    """

    def __init__(self, api_key: str, model: str = GROQ_MODEL):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        self.model = model
        print(f"  Using Groq model: {model}")

    def invoke(self, messages) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=_to_message_dicts(messages),
            max_tokens=1024,
            temperature=0.1,
        )
        return LLMResponse(response.choices[0].message.content)


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 4.  HuggingFace wrapper (fallback — uses metered credits)       ║
# ╚══════════════════════════════════════════════════════════════════╝
class HFLLMWrapper:
    """
    Wraps HF Inference Providers with the same .invoke(messages) interface.

    NOTE: free HF accounts get ~$0.10/month of Inference Provider credits.
    With no provider= argument the client routes via provider="auto" (often
    to a paid third party such as Novita) and bills those credits. Expect
    402 Payment Required once they're gone.
    """

    def __init__(self, token: str, model: str = HF_MODEL):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(model=model, token=token)
        self.model = model
        print(f"  Using HuggingFace model: {model}")

    def invoke(self, messages) -> LLMResponse:
        response = self.client.chat_completion(
            messages=_to_message_dicts(messages),
            max_tokens=1024,
            temperature=0.1,
        )
        return LLMResponse(response.choices[0].message.content)


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 5.  Provider factory — returns the right LLM backend            ║
# ╚══════════════════════════════════════════════════════════════════╝
def get_llm():
    """
    Return an LLM instance based on the environment.

    Precedence:
      1. GROQ_API_KEY → Groq                   (free tier, recommended)
      2. HF_TOKEN     → HF Inference Providers (needs credits)
      3. neither      → Ollama local model     (Codespaces / laptop)
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")

    if groq_key:
        print("LLM Provider: Groq (cloud)")
        return GroqLLMWrapper(api_key=groq_key)

    if hf_token:
        print("LLM Provider: HuggingFace Inference Providers (cloud)")
        return HFLLMWrapper(token=hf_token)

    print("LLM Provider: Ollama (local)")
    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL, temperature=0.0)


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 6.  Quick self-test                                              ║
# ╚══════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    print("=" * 50)
    print("LLM Provider — Self Test")
    print("=" * 50)
    llm = get_llm()
    print("\nSending test message...")
    response = llm.invoke([{"role": "user", "content": "Say hello in one sentence."}])
    print(f"Response: {response.content}")
