# Dev Container Image Maintenance

The Codespaces dev container uses a pre-built Docker image hosted on GitHub Container Registry (GHCR) to avoid large downloads at startup. When certain files change, the image must be rebuilt and pushed.

## Image location

```
ghcr.io/skillrepos/ai-apps-v3-devcontainer:latest
```

## What's baked into the image

| Component | How it got there |
|-----------|-----------------|
| Python 3 + venv | `apt-get install` |
| `/opt/py_env/` (full virtualenv) | `pip install -r requirements.txt` + CPU-only PyTorch swap |
| sentence-transformers embedding model | Pre-downloaded at build time |
| Ollama binary | Installed via `ollama.com/install.sh` (no model pulled) |
| zstd, curl, ca-certificates | `apt-get install` |

At codespace startup, `scripts/pysetup.sh` copies `/opt/py_env` to the workspace and fixes paths. No network downloads are needed.

## When to rebuild the image

Rebuild whenever you change any of these files:

| File changed | Why rebuild is needed |
|---|---|
| `requirements.txt` | Python packages are pre-installed in `/opt/py_env` |
| `.devcontainer/Dockerfile` | Image build instructions changed |
| Embedding model (in Dockerfile) | Model is pre-cached in the image |

Changes to these files do **not** require a rebuild:

- `.devcontainer/devcontainer.json` (VS Code settings, extensions, postCreateCommand)
- `scripts/pysetup.sh` (runs at startup, not at image build)
- `scripts/startOllama.sh` (runs manually by users)
- Any Python source files (`.py`), data files, or docs

## How to rebuild and push

Run from the repo root (the `COPY requirements.txt` in the Dockerfile needs the repo root as build context):

```bash
# 1. Log in to GHCR (one-time, or when token expires)
docker login ghcr.io -u YOUR_GITHUB_USERNAME

# 2. Build the image
docker build -f .devcontainer/Dockerfile -t ghcr.io/skillrepos/ai-apps-v3-devcontainer:latest .

# 3. Push to GHCR
docker push ghcr.io/skillrepos/ai-apps-v3-devcontainer:latest
```

Then commit and push your code changes (e.g. updated `requirements.txt`) so the repo and image stay in sync.

## Verifying the image is public

The GHCR package must be public for Codespaces to pull it without extra auth.

Check at: `github.com/users/skillrepos/packages/container/ai-apps-v3-devcontainer/settings`

Under "Danger Zone", visibility should be set to **Public**.

## Testing after a rebuild

1. Create a new codespace from the `main` branch
2. Confirm `py_env/` is populated and no `pip install` runs
3. Confirm `which ollama` returns `/usr/local/bin/ollama`
4. Confirm `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"` loads instantly (no download)
