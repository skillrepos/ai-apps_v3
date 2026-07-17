#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Prepare files for Hugging Face Spaces Deployment
# ═══════════════════════════════════════════════════════════════════════════════
#
# This script copies only the necessary files for deploying the AI Office
# Assistant to Hugging Face Spaces.
#
# Usage:
#   ./scripts/prepare_hf_spaces.sh [output_directory]
#
# If no output directory is specified, it defaults to the current directory.
#
# Files included:
#   - gradio_app.py       (Gradio UI — HF Spaces entry point)
#   - rag_agent.py         (Self-contained agent)
#   - llm_provider.py     (LLM backend provider)
#   - guardrails.py       (Prompt-injection detection)
#   - mcp_server.py       (MCP weather/geocoding/RAG tools)
#   - mcp_stdio_wrapper.py (Starts MCP server in stdio transport mode)
#   - data/offices.pdf    (Source PDF — indexed into ChromaDB on first run)
#   - requirements.txt    (Python dependencies for HF Spaces)
#   - README.md           (HF Spaces metadata and description)
#   - .gitignore          (Git ignore rules)
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Output directory (from argument, default to current directory)
OUTPUT_DIR="${1:-.}"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Preparing Hugging Face Spaces Deployment${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Source:${NC} $PROJECT_ROOT"
echo -e "${YELLOW}Output:${NC} $OUTPUT_DIR"
echo ""

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Copy Python source files
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${GREEN}Copying Python source files...${NC}"

cp "$PROJECT_ROOT/gradio_app.py" "$OUTPUT_DIR/"
cp "$PROJECT_ROOT/rag_agent.py" "$OUTPUT_DIR/"
cp "$PROJECT_ROOT/llm_provider.py" "$OUTPUT_DIR/"
cp "$PROJECT_ROOT/guardrails.py" "$OUTPUT_DIR/"
cp "$PROJECT_ROOT/mcp_server.py" "$OUTPUT_DIR/"
cp "$PROJECT_ROOT/mcp_stdio_wrapper.py" "$OUTPUT_DIR/"

# ─────────────────────────────────────────────────────────────────────────────
# Copy PDF data (the MCP server indexes it on first run)
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${GREEN}Copying PDF data...${NC}"

mkdir -p "$OUTPUT_DIR/data"
if [ -f "$PROJECT_ROOT/data/offices.pdf" ]; then
    cp "$PROJECT_ROOT/data/offices.pdf" "$OUTPUT_DIR/data/"
    echo "  Copied data/offices.pdf"
else
    echo -e "${YELLOW}  Warning: data/offices.pdf not found.${NC}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Create minimal requirements.txt for HF Spaces
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${GREEN}Creating requirements.txt...${NC}"

cat > "$OUTPUT_DIR/requirements.txt" << 'EOF'
# Hugging Face Spaces requirements for AI Office Assistant
# Note: gradio is pre-installed on HF Spaces with Gradio SDK

# Groq is OpenAI-compatible — the openai SDK is our cloud LLM client
openai>=1.60.0

# HuggingFace Hub (fallback LLM backend via Inference Providers)
huggingface_hub>=0.20.0

# Vector database for RAG
chromadb>=1.0.0

# Embeddings model
sentence-transformers>=5.0.0

# PDF text extraction (for on-the-fly indexing)
pdfplumber>=0.10.0

# MCP framework (server + stdio transport)
fastmcp>=2.0.0

# HTTP requests for weather APIs
requests>=2.31.0
EOF

# ─────────────────────────────────────────────────────────────────────────────
# Create HF Spaces README.md with metadata
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${GREEN}Creating README.md with HF Spaces metadata...${NC}"

cat > "$OUTPUT_DIR/README.md" << 'EOF'
---
title: AI Office Assistant
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: gradio_app.py
pinned: false
license: mit
---

# AI Office Assistant

An AI-powered office information agent built with:
- **Gradio** for the web interface
- **ChromaDB** for vector-based document search (RAG)
- **Groq** for LLM responses (free tier, OpenAI-compatible API)
- **Open-Meteo API** for live weather data

## IMPORTANT: Required Setup

**This Space requires `GROQ_API_KEY` to be set as a secret!**

Without `GROQ_API_KEY`, the LLM will not work.

**How to set it up:**
1. Get a free API key at [https://console.groq.com/keys](https://console.groq.com/keys) (no credit card needed)
2. Go to your Space's **Settings** page
3. Scroll to **Variables and secrets**
4. Click **New secret**
5. Name: `GROQ_API_KEY` / Value: Your Groq API key
6. Click **Save** — the Space will restart automatically

> Note: do **not** set `HF_TOKEN` as a secret here. A free Hugging Face account
> only gets ~$0.10/month of Inference Provider credits, which this agent
> exhausts after a couple of queries (`402 Payment Required`). Your HF token is
> only needed to `git push` to this Space, not to run it.

## Features

- RAG-powered office information retrieval
- Live weather data for office locations
- Natural language summaries with city facts
- Professional chat interface

## Example Queries

- "Tell me about HQ"
- "Tell me about the Southern office"
- "West Coast office details"

---
*Built for AI for App Development Workshop*
EOF

# ─────────────────────────────────────────────────────────────────────────────
# Create .gitignore
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${GREEN}Creating .gitignore...${NC}"

cat > "$OUTPUT_DIR/.gitignore" << 'EOF'
__pycache__/
*.py[cod]
*.so
*.log
.vscode/
.idea/
.DS_Store
Thumbs.db
.gradio/
flagged/
EOF

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Deployment files prepared successfully!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Files in $OUTPUT_DIR:"
echo ""
ls -la "$OUTPUT_DIR"
echo ""
echo -e "${YELLOW}REMINDER: Make sure GROQ_API_KEY is set as a secret in your Space!${NC}"
echo ""
