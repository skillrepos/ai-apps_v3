#!/usr/bin/env python3
"""
Lab 8: Gradio Web Interface with Conversation Memory
═══════════════════════════════════════════════════════════════════════
Evolves the Lab 7 Gradio interface to support conversation memory,
diverse query types, and a richer set of example queries.

KEY CHANGES from Lab 7
----------------------
- gr.State for conversation memory (persists across queries per session)
- chat_handler passes memory to run_agent and receives updated memory
- Memory indicator in sidebar shows conversation context
- New example queries demonstrate diverse agent capabilities
- Updated "How It Works" section reflects conversational features

ARCHITECTURE
------------
  User types query → chat_handler() → run_agent(prompt, memory) →
  (response, updated_memory) → response displayed, memory state updated
"""

# ═══════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")
warnings.filterwarnings("ignore", message=".*no_silent_downcasting.*")
warnings.filterwarnings("ignore", message=".*copy keyword is deprecated.*")

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

.memory-badge {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #166534;
}
"""


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 2.  Chat handler — bridges the UI to the agent (with memory)   ║
# ╚══════════════════════════════════════════════════════════════════╝

def chat_handler(message: str, history: list, memory: list) -> tuple:
    """
    Process a user message through the agent and update chat history.

    Parameters
    ----------
    message : str
        The user's input text
    history : list
        Gradio Chatbot history (list of {"role": ..., "content": ...} dicts)
    memory : list
        Conversation memory — list of (question, answer) tuples

    Returns
    -------
    tuple of (history, "", memory, memory_html)
        Updated history, empty string to clear input, updated memory,
        and HTML for the memory indicator
    """
    if not message.strip():
        return history, "", memory, _memory_html(memory)

    # Add user message to history
    history = history + [{"role": "user", "content": message}]

    if not AGENT_AVAILABLE:
        history = history + [{"role": "assistant",
                              "content": "[Demo Mode] Agent not available. "
                                         "Please ensure rag_agent.py is complete."}]
        return history, "", memory, _memory_html(memory)

    # Run the agent with conversation memory
    try:
        result, memory = run_agent(message, memory=memory)
        history = history + [{"role": "assistant", "content": result}]
    except Exception as e:
        history = history + [{"role": "assistant",
                              "content": f"Error processing query: {e}"}]

    return history, "", memory, _memory_html(memory)


def _memory_html(memory: list) -> str:
    """Generate HTML for the memory indicator in the sidebar."""
    count = len(memory)
    if count == 0:
        return ('<div class="memory-badge">'
                '<strong>Memory:</strong> No conversation history yet. '
                'Ask a question to get started!</div>')

    items = []
    for q, _a in memory[-5:]:
        items.append(f"<li>{q[:60]}{'...' if len(q) > 60 else ''}</li>")
    return (f'<div class="memory-badge">'
            f'<strong>Memory:</strong> {count} exchange(s)<br>'
            f'<ul style="margin:0.25rem 0 0 1rem;padding:0;">'
            f'{"".join(items)}</ul></div>')


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 3.  Gradio Interface Layout                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

with gr.Blocks(
    title="AI Office Assistant",
) as demo:

    # ── Persistent memory state (survives across queries) ────────
    memory_state = gr.State(value=[])

    # ── Header ─────────────────────────────────────────────────────
    gr.HTML("""
    <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%);
                padding: 1.5rem 2rem;
                border-radius: 14px;
                color: white;
                margin-bottom: 1rem;">
        <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700; color: #ffffff !important;">
            AI Office Assistant
        </h1>
        <p style="margin: 0.25rem 0 0 0; opacity: 0.9; font-size: 0.95rem; color: #e0e7ff !important;">
            Conversational Agent with Memory &middot; RAG + Live Weather
        </p>
    </div>
    """)

    # ── Main layout: chat area (left) + sidebar (right) ────────────
    with gr.Row():
        # ── Chat column ────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=450,
            )

            msg = gr.Textbox(
                placeholder="e.g. 'Tell me about HQ' then 'What services do they offer?'",
                show_label=False,
                container=False,
            )

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=3)
                clear_btn = gr.Button("Clear Chat & Memory", scale=1)

            gr.Markdown("**Try these queries (in order for follow-ups):**")
            with gr.Row():
                ex1 = gr.Button("Tell me about HQ", size="sm", variant="secondary")
                ex2 = gr.Button("What services do they offer?", size="sm", variant="secondary")
                ex3 = gr.Button("Which offices do Tech Development?", size="sm", variant="secondary")
                ex4 = gr.Button("Compare Tokyo and London", size="sm", variant="secondary")

        # ── Sidebar column ─────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Conversation Memory")
            memory_display = gr.HTML(
                '<div class="memory-badge">'
                '<strong>Memory:</strong> No conversation history yet. '
                'Ask a question to get started!</div>'
            )

            gr.Markdown("### Company Offices")
            gr.HTML("""
            <div class="office-card"><strong>HQ</strong> — New York, NY<br>200 employees · $15M revenue</div>
            <div class="office-card"><strong>West Coast Hub</strong> — San Francisco, CA<br>150 employees · $12M revenue</div>
            <div class="office-card"><strong>Southern</strong> — Austin, TX<br>80 employees · $5M revenue</div>
            <div class="office-card"><strong>Midwest</strong> — Chicago, IL<br>100 employees · $8M revenue</div>
            <div class="office-card"><strong>+ 16 more worldwide</strong></div>
            """)

            gr.Markdown("### How It Works")
            gr.Markdown(
                "1. **Smart Routing** — Agent decides which tools to use\n"
                "2. **RAG Search** — Finds office data in vector DB\n"
                "3. **Live Weather** — Fetches weather when relevant\n"
                "4. **Memory** — Remembers context for follow-ups\n"
                "5. **Summary** — LLM composes a conversational answer"
            )

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ 4.  Event handlers — wire UI actions to Python functions     ║
    # ╚══════════════════════════════════════════════════════════════╝

    # Send button and Enter key both trigger chat_handler
    send_btn.click(
        chat_handler,
        inputs=[msg, chatbot, memory_state],
        outputs=[chatbot, msg, memory_state, memory_display],
    )
    msg.submit(
        chat_handler,
        inputs=[msg, chatbot, memory_state],
        outputs=[chatbot, msg, memory_state, memory_display],
    )

    # Clear button resets the chat AND memory
    clear_btn.click(
        lambda: ([], "", [], '<div class="memory-badge">'
                 '<strong>Memory:</strong> No conversation history yet. '
                 'Ask a question to get started!</div>'),
        outputs=[chatbot, msg, memory_state, memory_display],
    )

    # Example buttons populate the input box
    ex1.click(lambda: "Tell me about HQ", outputs=[msg])
    ex2.click(lambda: "What services do they offer?", outputs=[msg])
    ex3.click(lambda: "Which offices do Tech Development?", outputs=[msg])
    ex4.click(lambda: "Compare the Tokyo and London offices", outputs=[msg])

    # ── Footer ─────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align: center; padding: 1rem; margin-top: 1rem;
                border-top: 1px solid #e2e8f0; color: #94a3b8; font-size: 0.8rem;">
        AI Office Assistant: Agents, RAG and Local Models
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
    print("AI Office Assistant — Conversational Interface")
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
