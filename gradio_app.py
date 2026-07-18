#!/usr/bin/env python3
"""
Lab 7: Gradio Web Interface for the Office Agent
═══════════════════════════════════════════════════════════════════════
Adds a professional web UI on top of the self-contained agent from Lab 6.

KEY CONCEPTS
------------
- gr.Blocks: Gradio's flexible layout system (rows, columns, components)
- gr.Chatbot: Built-in chat component with message history
- Event handlers: .click() and .submit() wire UI actions to Python functions
- gr.themes.Soft(): Pre-built professional theme

ARCHITECTURE
------------
  User types query → chat_handler() → rag_agent.run_agent() → response
  Response displayed in Chatbot component with full conversation history
"""

# ═══════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════

import gradio as gr

# ─────────────────────────────────────────────────────────────────────
# Agent Import — if the agent isn't available, the UI runs in demo mode
# ─────────────────────────────────────────────────────────────────────
try:
    from rag_agent import run_agent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("Warning: rag_agent not available. Running in demo mode.")


# ─────────────────────────────────────────────────────────────────────
# ZeroGPU startup probe
# ─────────────────────────────────────────────────────────────────────
# Hugging Face moved FREE Gradio Spaces to ZeroGPU-only (CPU Basic now needs
# PRO). ZeroGPU refuses to start unless it detects a @spaces.GPU function —
# otherwise you get "No @spaces.GPU function detected during startup".
# This app does ALL its work on CPU + the Groq cloud API, so we register a
# no-op probe purely to satisfy that check. It is never called, so it uses
# none of the free daily GPU quota.
# The `spaces` package only exists on HF Spaces; locally (Lab 7 in the
# codespace) we fall back to a stub so @spaces.GPU is a plain passthrough.
try:
    import spaces
except ImportError:
    class _SpacesStub:
        @staticmethod
        def GPU(func=None, **kwargs):
            return func if func is not None else (lambda f: f)
    spaces = _SpacesStub()


@spaces.GPU
def _zerogpu_startup_probe():
    """No-op — exists only so HF ZeroGPU detects a GPU function at startup."""
    return None


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 1.  Custom CSS for professional styling                         ║
# ╚══════════════════════════════════════════════════════════════════╝

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.office-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.375rem 0;
    font-size: 0.9rem;
    color: #475569;
}
"""


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 2.  Chat handler — bridges the UI to the agent                  ║
# ╚══════════════════════════════════════════════════════════════════╝

# TODO: Implement chat_handler(message, history) that:
#       1. Adds the user message to history
#       2. Calls run_agent(message) to get a response
#       3. Adds the assistant response to history
#       4. Returns updated history and empty string to clear input


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 3.  Gradio Interface Layout                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

with gr.Blocks(
    title="AI Office Assistant",
) as demo:

    # ── Header ─────────────────────────────────────────────────────
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
                padding: 1.5rem 2rem;
                border-radius: 14px;
                color: white;
                margin-bottom: 1rem;">
        <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700;">
            AI Office Assistant
        </h1>
        <p style="margin: 0.25rem 0 0 0; opacity: 0.9; font-size: 0.95rem;">
            Powered by RAG + Live Weather Data
        </p>
    </div>
    """)

    # ── Main layout: chat area (left) + sidebar (right) ────────────
    with gr.Row():
        # ── Chat column ────────────────────────────────────────────
        with gr.Column(scale=3):
            # TODO: Add Chatbot component, Textbox input, Send/Clear buttons,
            #       and example query buttons
            pass  # Remove after merging

        # ── Sidebar column ─────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Company Offices")
            gr.HTML("""
            <div class="office-card"><strong>HQ</strong> — New York, NY<br>150 employees · $85.5M revenue</div>
            <div class="office-card"><strong>West Coast</strong> — San Francisco, CA<br>95 employees · $78.9M revenue</div>
            <div class="office-card"><strong>Southern</strong> — Austin, TX<br>80 employees · $52.3M revenue</div>
            <div class="office-card"><strong>Midwest</strong> — Chicago, IL<br>120 employees · $67.2M revenue</div>
            <div class="office-card"><strong>Southeast</strong> — Atlanta, GA<br>75 employees · $48.7M revenue</div>
            """)

            gr.Markdown("### How It Works")
            gr.Markdown(
                "1. **RAG Search** — Finds office data in vector DB\n"
                "2. **Geocoding** — Looks up city coordinates\n"
                "3. **Weather** — Fetches live weather data\n"
                "4. **Summary** — LLM composes a friendly answer"
            )

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ 4.  Event handlers — wire UI actions to Python functions     ║
    # ╚══════════════════════════════════════════════════════════════╝

    # TODO: Wire send button, Enter key, clear button, and example
    #       buttons to their handler functions

    # ── Footer ─────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align: center; padding: 1rem; margin-top: 1rem;
                border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 0.8rem;">
        AI Office Assistant
        · <a href="https://getskillsnow.com" target="_blank"
              style="color: #64748b; text-decoration: none;">
            &copy; 2026 Tech Skills Transformations
          </a>
    </div>
    """)


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 5.  Main entry point                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    print("=" * 60)
    print("AI Office Assistant — Gradio Interface")
    print("=" * 60)
    print(f"Agent available: {AGENT_AVAILABLE}")
    print("Starting Gradio interface...")
    print("=" * 60)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
    )
