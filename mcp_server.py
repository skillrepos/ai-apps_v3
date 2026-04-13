#!/usr/bin/env python3
"""
Lab 8: MCP Server with RAG + Service & Overview Tools (Conversational)
────────────────────────────────────────────────────────────────────────
Evolves the Lab 6 MCP server to support richer agentic queries by adding
tools for service-based search and office overview.

Tools Provided
--------------
1. get_weather(lat, lon) → dict with temperature °C, WMO code, conditions
2. convert_c_to_f(c) → float (temperature in °F)
3. geocode_location(name) → dict with latitude, longitude, location name
4. search_offices(query) → text chunks from office vector DB
5. find_offices_by_service(service) → offices offering a specific service (NEW)
6. list_all_offices() → all office entries for overview queries (NEW)

Key Changes from Lab 6
----------------------
- Adds find_offices_by_service() for service-focused queries like
  "Which offices do Tech Development?" — returns more results (8) for
  broader coverage across the office database
- Adds list_all_offices() for overview and comparison queries like
  "How many offices are there?" — returns all entries (up to 20)
- These new tools let the agent handle diverse question types without
  always following the same search → geocode → weather pattern
"""

from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────
import re
import time
from pathlib import Path
from typing import Final, List

# ── 3rd-party ───────────────────────────────────────────────────────
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import pdfplumber
import requests
from fastmcp import FastMCP

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 1.  Weather-code lookup table (WMO standard codes)               ║
# ╚══════════════════════════════════════════════════════════════════╝
# Open-Meteo returns WMO weather codes - this maps them to descriptions.
# WMO (World Meteorological Organization) codes are used by many weather APIs.
WEATHER_CODES: Final[dict[int, str]] = {
    0:  "Clear sky",                     1:  "Mainly clear",
    2:  "Partly cloudy",                 3:  "Overcast",
    45: "Fog",                           48: "Depositing rime fog",
    51: "Light drizzle",                 53: "Moderate drizzle",
    55: "Dense drizzle",                 56: "Light freezing drizzle",
    57: "Dense freezing drizzle",        61: "Slight rain",
    63: "Moderate rain",                 65: "Heavy rain",
    66: "Light freezing rain",           67: "Heavy freezing rain",
    71: "Slight snow fall",              73: "Moderate snow fall",
    75: "Heavy snow fall",               77: "Snow grains",
    80: "Slight rain showers",           81: "Moderate rain showers",
    82: "Violent rain showers",          85: "Slight snow showers",
    86: "Heavy snow showers",            95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 2.  Retry configuration for API resilience                       ║
# ╚══════════════════════════════════════════════════════════════════╝
# Shared retry settings for all external API calls
MAX_RETRIES    = 3       # Total attempts (1 original + 2 retries)
BACKOFF_FACTOR = 1.5     # Exponential backoff: 1.5s, 2.25s, 3.375s
TRANSIENT_CODES = {429, 500, 502, 503, 504}  # HTTP codes worth retrying

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 3.  ChromaDB setup for office RAG search                          ║
# ╚══════════════════════════════════════════════════════════════════╝
CHROMA_PATH     = Path(__file__).parent / "chroma_db"
PDF_DIR         = Path(__file__).parent / "data"
COLLECTION_NAME = "codebase"
TOP_K           = 3

embed_fn = DefaultEmbeddingFunction()

# ── Regex for splitting PDF text into lines ──────────────────────────
LINE_RE = re.compile(r"[^\S\r\n]*\r?\n[^\S\r\n]*")

def _extract_lines(path: Path) -> List[str]:
    """Extract every non-blank line from a PDF file."""
    lines: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw_line in LINE_RE.split(text):
                line = raw_line.strip()
                if line:
                    lines.append(line)
    return lines

def _build_index(coll: chromadb.Collection) -> None:
    """Index all PDFs in data/ into the ChromaDB collection."""
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        print(f"  Indexing {pdf_path.name}...")
        for idx, line in enumerate(_extract_lines(pdf_path)):
            coll.add(
                ids=[f"{pdf_path.name}-{idx}"],
                documents=[line],
                metadatas=[{"path": str(pdf_path), "chunk_index": idx}],
            )
    print(f"  Indexed {coll.count()} chunks.")

def open_collection() -> chromadb.Collection:
    """Open the ChromaDB collection, building the index if empty."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    coll = client.get_or_create_collection(COLLECTION_NAME)
    if coll.count() == 0:
        print("ChromaDB empty — building index from PDFs...")
        _build_index(coll)
    return coll

coll = open_collection()

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 4.  MCP Server initialization and tool definitions               ║
# ╚══════════════════════════════════════════════════════════════════╝
mcp = FastMCP("WeatherServer")

# ─── Office Search Tool ─────────────────────────────────────────────

@mcp.tool
def search_offices(query: str) -> str:
    """
    Search the office vector database for relevant information.
    This is the 'Retrieval' part of RAG — semantic search over
    the office PDF data indexed in Lab 4.

    Parameters
    ----------
    query : str
        Natural language search query (e.g., "HQ", "Southern office")

    Returns
    -------
    str
        Top matching text chunks, separated by '---'
    """
    query_vec = embed_fn([query])[0]
    res = coll.query(
        query_embeddings=[query_vec],
        n_results=TOP_K,
        include=["documents"],
    )
    docs = res["documents"][0] if res["documents"] else []
    if not docs:
        return "No matching office information found."
    return "\n---\n".join(docs)

# ─── Service Search Tool (NEW in Lab 8) ─────────────────────────────

@mcp.tool
def find_offices_by_service(service: str) -> str:
    """
    Find offices that offer a specific service type.

    Searches the office database for entries matching the given service
    (e.g., "Tech Development", "Marketing", "Customer Support").
    Returns more results (8) than search_offices for broader coverage.

    Parameters
    ----------
    service : str
        Service type to search for (e.g., "Tech Development", "Sales")

    Returns
    -------
    str
        Matching office entries, separated by '---'
    """
    query_vec = embed_fn([f"offices offering {service}"])[0]
    res = coll.query(
        query_embeddings=[query_vec],
        n_results=8,
        include=["documents"],
    )
    docs = res["documents"][0] if res["documents"] else []
    if not docs:
        return f"No offices found offering {service}."
    return "\n---\n".join(docs)

# ─── List All Offices Tool (NEW in Lab 8) ───────────────────────────

@mcp.tool
def list_all_offices() -> str:
    """
    List all offices in the database with their basic information.

    Returns a broad set of office entries for overview or comparison
    queries. Useful when the user asks "how many offices" or wants
    to see all locations.

    Returns
    -------
    str
        All office entries, separated by '---'
    """
    query_vec = embed_fn(["office location employees revenue services"])[0]
    res = coll.query(
        query_embeddings=[query_vec],
        n_results=20,
        include=["documents"],
    )
    docs = res["documents"][0] if res["documents"] else []
    if not docs:
        return "No office information found."
    return "\n---\n".join(docs)

# ─── Weather Tool ────────────────────────────────────────────────────

@mcp.tool
def get_weather(lat: float, lon: float) -> dict:
    """
    Fetch **current weather** from Open-Meteo and return a concise dict.

    Retry policy
    ------------
    * Up to MAX_RETRIES total attempts with fresh connections.
    * Retries on network errors **or** HTTP 429/5xx.
    * Exponential back-off (1.5 s, 2.25 s, …).
    * Each retry uses a new session to avoid connection pool issues.

    Parameters
    ----------
    lat, lon : float
        Geographic coordinates in decimal degrees.

    Returns
    -------
    dict
        {
            "temperature": <float °C>,
            "code":        <int WMO weathercode>,
            "conditions":  <friendly description>,
            "error":       <error message if request failed>
        }
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )

    last_error = None

    # Retry loop with fresh connections
    for attempt in range(MAX_RETRIES):
        try:
            # Fresh session per attempt avoids connection pool reuse issues
            session = requests.Session()
            resp = session.get(url, timeout=15)
            session.close()

            # Handle rate limiting and server errors with retry
            if resp.status_code in TRANSIENT_CODES:
                last_error = f"HTTP {resp.status_code}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_FACTOR ** attempt)
                    continue

            resp.raise_for_status()

            # Extract and return weather data
            cw = resp.json()["current_weather"]
            code = cw["weathercode"]
            return {
                "temperature": cw["temperature"],
                "code":        code,
                "conditions":  WEATHER_CODES.get(code, "Unknown"),
            }

        except requests.HTTPError as e:
            # HTTP errors (4xx, 5xx not already caught)
            last_error = f"HTTP {e.response.status_code}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** attempt)
                continue

        except requests.RequestException as e:
            # Network errors (timeout, connection refused, etc.)
            last_error = f"{type(e).__name__}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** attempt)
                continue

        except (KeyError, ValueError) as e:
            # Data format errors - don't retry, immediate failure
            return {
                "error": f"Received invalid data from weather service: {type(e).__name__}. Please try again later."
            }

    # All retries exhausted - return graceful error
    return {
        "error": f"Weather service failed after {MAX_RETRIES} attempts (last error: {last_error}). Please try again later."
    }


# ─── Temperature Conversion Tool ─────────────────────────────────────

@mcp.tool
def convert_c_to_f(c: float) -> float:
    """Simple Celsius → Fahrenheit conversion."""
    return c * 9 / 5 + 32


# ─── Geocoding Tool ──────────────────────────────────────────────────

@mcp.tool
def geocode_location(name: str) -> dict:
    """
    Geocode a location name to latitude/longitude coordinates using Open-Meteo's geocoding API.

    Retry policy
    ------------
    * Up to MAX_RETRIES total attempts with fresh connections.
    * Retries on network errors **or** HTTP 429/5xx.
    * Exponential back-off (1.5 s, 2.25 s, …).
    * Each retry uses a new session to avoid connection pool issues.

    Parameters
    ----------
    name : str
        Location name (e.g., "San Francisco", "Paris, France", "London, UK")

    Returns
    -------
    dict
        {
            "latitude": <float>,
            "longitude": <float>,
            "name": <matched location name>,
            "error": <error message if request failed>
        }
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    last_error = None

    # Retry loop with fresh connections
    for attempt in range(MAX_RETRIES):
        try:
            # Fresh session per attempt avoids connection pool reuse issues
            session = requests.Session()
            resp = session.get(url, params={"name": name, "count": 1}, timeout=15)
            session.close()

            # Handle rate limiting and server errors with retry
            if resp.status_code in TRANSIENT_CODES:
                last_error = f"HTTP {resp.status_code}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_FACTOR ** attempt)
                    continue

            resp.raise_for_status()

            # Parse and return geocoding results
            data = resp.json()
            if data.get("results"):
                hit = data["results"][0]
                return {
                    "latitude": hit["latitude"],
                    "longitude": hit["longitude"],
                    "name": hit.get("name", name),
                }
            else:
                # No results found - not an error, just no match
                return {
                    "error": f"No location found for '{name}'. Try a different search term."
                }

        except requests.HTTPError as e:
            # HTTP errors (4xx, 5xx not already caught)
            last_error = f"HTTP {e.response.status_code}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** attempt)
                continue

        except requests.RequestException as e:
            # Network errors (timeout, connection refused, etc.)
            last_error = f"{type(e).__name__}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** attempt)
                continue

        except (KeyError, ValueError) as e:
            # Data format errors - don't retry, immediate failure
            return {
                "error": f"Received invalid data from geocoding service: {type(e).__name__}. Please try again later."
            }

    # All retries exhausted - return graceful error
    return {
        "error": f"Geocoding service failed after {MAX_RETRIES} attempts (last error: {last_error}). Please try again later."
    }

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 5.  Server startup                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    # Start HTTP server using FastAPI + Uvicorn
    # Clients connect to: http://127.0.0.1:8000/mcp/
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8000,
        path="/mcp/",
    )
