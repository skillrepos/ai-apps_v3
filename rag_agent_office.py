#!/usr/bin/env python3
"""
Lab 6: RAG-Enhanced Office Agent (Deployable)
═══════════════════════════════════════════════════════════════════════
Evolves the RAG agent from Lab 5 to be deployment-ready:
- Uses llm_provider for flexible LLM backend (Ollama OR HF Inference)
- Starts the MCP server as a SUBPROCESS via stdio transport
- Adds guardrails for prompt-injection defence at three boundaries
- Ready for Gradio interface (Lab 7) and HF deployment (Lab 8)

Tools Available (all via MCP server)
------------------------------------
1. search_offices(query) → text chunks from office vector DB
2. geocode_location(name) → lat/lon coordinates
3. get_weather(lat, lon)  → current weather in Celsius
4. convert_c_to_f(c)      → temperature in Fahrenheit

Key Changes from Lab 5
------------------------------------------
- Lab 5 connected to the MCP server over HTTP → requires running it
  separately with "python mcp_server.py"
- This version starts the MCP server as a SUBPROCESS and connects
  via stdio transport → fully self-contained, one command to run
- Lab 5 used ChatOllama directly → this uses llm_provider (Ollama or HF)
- Adds guardrails at three boundaries: input, tool results, output
- Adds a synchronous wrapper so Gradio can call it easily
"""

# ────────────────────────── standard libs ───────────────────────────
import asyncio
import json
import re
import textwrap
from pathlib import Path

# ────────────────────────── third-party libs ────────────────────────
from fastmcp import Client

# ────────────────────────── our modules ─────────────────────────────
from llm_provider import get_llm
from guardrails import check_input, check_tool_result, check_output

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 1.  Configuration                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
MCP_ENDPOINT     = "http://127.0.0.1:8000/mcp/" # MCP server from Lab 5

# Regex for parsing LLM responses (same pattern as Labs 2 and 3)
ACTION_RE = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
ARGS_RE   = re.compile(r"Args:\s*(\{.*?\})(?:\s|$)", re.S | re.IGNORECASE)

# MCP server subprocess — starts mcp_server.py via stdio instead of
# connecting over HTTP. FastMCP's Client sees a .py path and auto-
# starts it as a child process, talking MCP over stdin/stdout.
MCP_SERVER = str(Path(__file__).parent / "mcp_stdio_wrapper.py")

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 2.  MCP result unwrapper                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
def unwrap(obj):
    """Extract plain Python values from FastMCP result wrappers."""
    if hasattr(obj, "structured_content") and obj.structured_content:
        return unwrap(obj.structured_content)
    if hasattr(obj, "data") and obj.data:
        return unwrap(obj.data)
    if hasattr(obj, "text"):
        try:
            return json.loads(obj.text)
        except Exception:
            return obj.text
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, list) and len(obj) == 1:
        return unwrap(obj[0])
    if isinstance(obj, dict):
        if len(obj) == 1:
            return unwrap(list(obj.values())[0])
        numeric_vals = [v for v in obj.values() if isinstance(v, (int, float))]
        if len(numeric_vals) == 1:
            return numeric_vals[0]
    return obj

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 3.  System prompt — tells the LLM about all available tools     ║
# ╚══════════════════════════════════════════════════════════════════╝
SYSTEM = textwrap.dedent("""
You are an office information agent. You answer questions about company
offices by searching a database and looking up live weather data.

You have these tools:

search_offices(query: str)
    Searches the company office database for matching information.
    Returns: text chunks with office names, cities, and details.
    ALWAYS call this first to find relevant office information.

geocode_location(name: str)
    Converts a city/location name to coordinates.
    Returns: {"latitude": float, "longitude": float, "name": str}

get_weather(lat: float, lon: float)
    Gets current weather for given coordinates.
    Returns: {"temperature": float, "code": int, "conditions": str}
    Note: temperature is in Celsius.

convert_c_to_f(c: float)
    Converts a Celsius temperature to Fahrenheit.
    Returns: float

IMPORTANT: Respond with EXACTLY ONE step at a time — a single
Thought/Action/Args triplet. Wait for the Observation before continuing.

Thought: <your reasoning about what to do next>
Action: <exact tool name only>
Args: <valid JSON arguments for the tool>

Examples:

Thought: I need to search for office information about HQ
Action: search_offices
Args: {"query": "HQ"}

Thought: I found the office is in Austin, TX. I need coordinates.
Action: geocode_location
Args: {"name": "Austin"}

Thought: Now I'll get the weather at those coordinates
Action: get_weather
Args: {"lat": 30.2672, "lon": -97.7431}

Thought: I need to convert 25.0 Celsius to Fahrenheit
Action: convert_c_to_f
Args: {"c": 25.0}

When you have gathered all the information, respond with:
Thought: I have all the information needed
Action: DONE
Args: {}
Final: <a friendly 2-3 sentence summary including the office name and
city, weather conditions and temperature in Fahrenheit, and one
interesting fact about the city>

Do NOT put arguments on the Action line. Arguments go ONLY in the Args line as JSON.

RULES:
1. ALWAYS start with search_offices to find office data
2. The FIRST search result is the most relevant — use the city from it
3. When geocoding, use ONLY the city name (e.g. "New York" not "New York, NY")
4. If geocoding fails, retry with a simpler name before trying other cities
5. Get the weather, then convert the temperature
6. Office details and weather MUST come from tool results — do NOT invent them
7. You may add one interesting fact about the city from your own knowledge
8. Do NOT add extra text beyond the required format
9. Respond with ONLY ONE Thought/Action/Args per message — NEVER plan ahead
""").strip()

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 4.  Async TAO agent loop (starts MCP server via stdio)          ║
# ╚══════════════════════════════════════════════════════════════════╝
async def _run_agent_async(prompt: str, max_steps: int = 10) -> str:
    """
    Run the TAO agent loop with the MCP server as a subprocess.

    The MCP server is started via stdio transport — the agent spawns
    mcp_stdio_wrapper.py as a child process and talks MCP protocol
    over stdin/stdout. ALL tools go through MCP — the agent is a
    pure orchestrator.
    """
    # ── Guardrail: check user input before the LLM sees it ───────
    is_safe, prompt = check_input(prompt)
    if not is_safe:
        print(f"\n⚠️  Prompt blocked by guardrails.")
        return prompt          # prompt now holds the refusal message

    llm = get_llm()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    print("\n" + "="*60)
    print("RAG Agent — Thought / Action / Observation")
    print("="*60 + "\n")

    # Start MCP server as subprocess and connect via stdio
    async with Client(MCP_SERVER) as mcp:
        for step in range(1, max_steps + 1):
            print(f"[Step {step}]")

            # Ask the LLM what to do next
            response = llm.invoke(messages).content.strip()
            print(response)

            # Parse the Action from the response
            action_match = ACTION_RE.search(response)
            if not action_match:
                print("\nError: Could not parse Action from response\n")
                break

            action = action_match.group(1).lower()

            # ── Check if the agent decided it is done ─────────────────
            if action == "done":
                # Some models include "Final:" in the DONE response;
                # others omit it.  If missing, ask the LLM once more.
                if "Final:" not in response:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content":
                        "Now provide your Final: summary."})
                    response = llm.invoke(messages).content.strip()
                    print(response)

                print("\n" + "="*60)
                if "Final:" in response:
                    final = response.split("Final:", 1)[1].strip()
                else:
                    # Model gave the summary as plain text
                    final = response.strip()

                # ── Guardrail: sanitise output before user sees it ─
                final = check_output(final)
                print(f"\n{final}\n")
                return final

            # ── Parse the Args ────────────────────────────────────────
            args_match = ARGS_RE.search(response)
            if not args_match:
                print("\nError: Could not parse Args from response\n")
                break

            try:
                args = json.loads(args_match.group(1))
            except json.JSONDecodeError as e:
                print(f"\nError: Invalid JSON: {e}\n")
                break

            # ── Call the tool via MCP ─────────────────────────────────
            print(f"\n-> Calling: {action}({json.dumps(args)})")

            try:
                raw = await mcp.call_tool(action, args)
                result = unwrap(raw)
            except Exception as e:
                result = f"Error: {type(e).__name__}: {e}"

            # Format the observation
            if isinstance(result, (dict, float, int)):
                obs_text = json.dumps(result)
            else:
                obs_text = str(result)

            # ── Guardrail: check tool result for injected content ──
            _safe, obs_text = check_tool_result(action, obs_text)
            if not _safe:
                print(f"⚠️  Tool result sanitised by guardrails.")

            print(f"Observation: {obs_text}\n")

            # Feed the observation back to the LLM.
            # Truncate to the first Thought/Action/Args triplet only —
            # some models (e.g. HF Inference) plan multiple steps at once
            # which confuses the loop if appended in full.
            first_step = response[:args_match.end()]
            messages.append({"role": "assistant", "content": first_step})
            messages.append({"role": "user",
                             "content": f"Observation: {obs_text}"})

    return "Reached maximum steps without completing."


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 5.  Synchronous wrapper (for Gradio and command-line use)       ║
# ╚══════════════════════════════════════════════════════════════════╝
def run_agent(prompt: str, max_steps: int = 10) -> str:
    """
    Synchronous entry point that Gradio and the command line use.
    Wraps the async agent loop with asyncio.run().
    """
    return asyncio.run(_run_agent_async(prompt, max_steps))


# ╔══════════════════════════════════════════════════════════════════╗
# ║ 6.  Interactive loop                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    print("=" * 60)
    print("Office Agent (Deployable Version)")
    print("=" * 60)
    print("\nAsk about any office (e.g. 'Tell me about HQ')")
    print("Type 'exit' to quit\n")

    while True:
        prompt = input("User: ").strip()
        if prompt.lower() == "exit":
            print("Goodbye!")
            break
        if prompt:
            run_agent(prompt)
            print()
