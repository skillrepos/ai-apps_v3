#!/usr/bin/env bash
#
# prebake-devcontainer.sh
# =======================
# Transforms a repo's devcontainer setup to use a pre-built Docker image
# hosted on GitHub Container Registry (GHCR). This eliminates large
# downloads at Codespaces startup — ideal for workshops, conferences,
# or any bandwidth-constrained environment.
#
# Usage:
#   cd /path/to/your/repo
#   bash prebake-devcontainer.sh [--ghcr-org ORG] [--image-name NAME]
#
# What it does:
#   1. Reads .devcontainer/devcontainer.json to detect current setup
#   2. Generates a Dockerfile that pre-bakes dependencies into the image
#   3. Updates devcontainer.json to reference the GHCR image
#   4. Generates IMAGE-MAINTENANCE.md with rebuild instructions
#   5. Prints build/push commands to run
#
# What it does NOT do:
#   - Build or push the Docker image (you run those commands yourself)
#   - Modify any application code or scripts
#   - Set up Codespaces prebuilds (that's a GitHub UI setting)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
GHCR_ORG=""
IMAGE_NAME=""
REPO_ROOT="$(pwd)"
DEVCONTAINER_DIR="$REPO_ROOT/.devcontainer"
DEVCONTAINER_JSON="$DEVCONTAINER_DIR/devcontainer.json"

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --ghcr-org)  GHCR_ORG="$2"; shift 2 ;;
        --image-name) IMAGE_NAME="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash prebake-devcontainer.sh [--ghcr-org ORG] [--image-name NAME]"
            echo ""
            echo "Options:"
            echo "  --ghcr-org ORG      GitHub org or username for GHCR (default: from git remote)"
            echo "  --image-name NAME   Image name on GHCR (default: <repo>-devcontainer)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Preflight checks ─────────────────────────────────────────────────
if [[ ! -f "$DEVCONTAINER_JSON" ]]; then
    echo "ERROR: No .devcontainer/devcontainer.json found in $(pwd)"
    echo "Run this script from the root of a repo with a devcontainer setup."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required (used for JSON parsing)."
    exit 1
fi

# ── Derive GHCR org and image name from git remote if not provided ────
if [[ -z "$GHCR_ORG" || -z "$IMAGE_NAME" ]]; then
    REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
    if [[ -n "$REMOTE_URL" ]]; then
        # Extract org/repo from HTTPS or SSH remote URLs
        REPO_SLUG=$(echo "$REMOTE_URL" | sed -E 's#.+github\.com[:/]##; s/\.git$//')
        GIT_ORG=$(echo "$REPO_SLUG" | cut -d/ -f1)
        GIT_REPO=$(echo "$REPO_SLUG" | cut -d/ -f2)
        [[ -z "$GHCR_ORG" ]] && GHCR_ORG="$GIT_ORG"
        [[ -z "$IMAGE_NAME" ]] && IMAGE_NAME="${GIT_REPO}-devcontainer"
    fi
fi

if [[ -z "$GHCR_ORG" || -z "$IMAGE_NAME" ]]; then
    echo "ERROR: Could not determine GHCR org/image name from git remote."
    echo "Provide them explicitly: --ghcr-org ORG --image-name NAME"
    exit 1
fi

FULL_IMAGE="ghcr.io/$GHCR_ORG/$IMAGE_NAME:latest"
echo ""
echo "============================================================"
echo "  prebake-devcontainer"
echo "============================================================"
echo "  Repo root:    $REPO_ROOT"
echo "  GHCR image:   $FULL_IMAGE"
echo "============================================================"
echo ""

# ── Detect current setup ─────────────────────────────────────────────
echo "Analyzing .devcontainer/devcontainer.json..."

# Use python3 for reliable JSON parsing (handles comments by stripping them)
PARSED=$(python3 -c "
import json, re, sys

with open('$DEVCONTAINER_JSON') as f:
    text = f.read()

# Strip JS-style comments (// ...) that devcontainer.json allows
text = re.sub(r'//.*', '', text)

data = json.loads(text)

print('BASE_IMAGE=' + json.dumps(data.get('image', '')))
print('HAS_BUILD=' + ('1' if 'build' in data else '0'))
print('FEATURES=' + json.dumps(list(data.get('features', {}).keys())))
print('POST_CREATE=' + json.dumps(data.get('postCreateCommand', '')))
print('POST_START=' + json.dumps(data.get('postStartCommand', '')))
print('ON_CREATE=' + json.dumps(data.get('onCreateCommand', '')))
")

eval "$PARSED"

echo "  Base image:        ${BASE_IMAGE:-<none - uses build>}"
echo "  Has Dockerfile:    $([ "$HAS_BUILD" = "1" ] && echo "yes" || echo "no")"
echo "  Features:          $FEATURES"
echo "  postCreateCommand: $POST_CREATE"
echo ""

# ── Detect dependencies ──────────────────────────────────────────────
echo "Scanning for dependency files..."

HAS_PYTHON_REQS=0
HAS_PYPROJECT=0
HAS_PIPFILE=0
HAS_PACKAGE_JSON=0
HAS_GEMFILE=0
PYTHON_FEATURE=0
NODE_FEATURE=0
HAS_PYTORCH=0

[[ -f "$REPO_ROOT/requirements.txt" ]] && HAS_PYTHON_REQS=1 && echo "  Found: requirements.txt"
[[ -f "$REPO_ROOT/pyproject.toml" ]]   && HAS_PYPROJECT=1    && echo "  Found: pyproject.toml"
[[ -f "$REPO_ROOT/Pipfile" ]]          && HAS_PIPFILE=1       && echo "  Found: Pipfile"
[[ -f "$REPO_ROOT/package.json" ]]     && HAS_PACKAGE_JSON=1  && echo "  Found: package.json"
[[ -f "$REPO_ROOT/Gemfile" ]]          && HAS_GEMFILE=1        && echo "  Found: Gemfile"

echo "$FEATURES" | grep -qi "python" && PYTHON_FEATURE=1
echo "$FEATURES" | grep -qi "node"   && NODE_FEATURE=1

# Check if PyTorch is a dependency
if [[ $HAS_PYTHON_REQS -eq 1 ]] && grep -qi "torch" "$REPO_ROOT/requirements.txt"; then
    HAS_PYTORCH=1
    echo "  Detected: PyTorch dependency"
fi

echo ""

# ── Detect postCreateCommand scripts that may re-download deps ────────
echo "Checking postCreateCommand for potential re-download issues..."
POST_CREATE_CLEAN=$(echo "$POST_CREATE" | tr -d '"')
HAS_PIP_IN_POST=0
if [[ -n "$POST_CREATE_CLEAN" ]]; then
    # Find scripts referenced in postCreateCommand
    for script in $(echo "$POST_CREATE_CLEAN" | grep -oE '[^ ]+\.sh'); do
        script_path="$REPO_ROOT/$script"
        if [[ -f "$script_path" ]]; then
            # Check if the script runs pip install outside of a conditional block
            if grep -qE 'pip.*install.*(--upgrade|torch|requirements)' "$script_path" 2>/dev/null; then
                HAS_PIP_IN_POST=1
                echo ""
                echo "  WARNING: $script contains pip install commands that may"
                echo "  re-download packages already baked into the image."
                echo "  Ensure those commands only run in a fallback path"
                echo "  (e.g., inside an 'else' block when /opt/py_env is missing)."
                echo ""
            fi
        fi
    done
fi
if [[ $HAS_PIP_IN_POST -eq 0 ]]; then
    echo "  No issues detected."
fi
echo ""

# ── Determine base image for Dockerfile ──────────────────────────────
# Use the existing base image, or a sensible default
DOCKERFILE_BASE="$BASE_IMAGE"
if [[ -z "$DOCKERFILE_BASE" || "$DOCKERFILE_BASE" == '""' ]]; then
    DOCKERFILE_BASE="mcr.microsoft.com/devcontainers/base:bookworm"
fi
# Strip surrounding quotes from JSON
DOCKERFILE_BASE=$(echo "$DOCKERFILE_BASE" | tr -d '"')

# ── Generate Dockerfile ──────────────────────────────────────────────
DOCKERFILE="$DEVCONTAINER_DIR/Dockerfile"
BACKUP_SUFFIX=".backup.$(date +%Y%m%d%H%M%S)"

if [[ -f "$DOCKERFILE" ]]; then
    cp "$DOCKERFILE" "${DOCKERFILE}${BACKUP_SUFFIX}"
    echo "Backed up existing Dockerfile to Dockerfile${BACKUP_SUFFIX}"
fi

echo "Generating .devcontainer/Dockerfile..."

cat > "$DOCKERFILE" << DOCKERFILE_EOF
FROM $DOCKERFILE_BASE

# Install common prerequisites
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
        curl ca-certificates && \\
    rm -rf /var/lib/apt/lists/*
DOCKERFILE_EOF

# Add Python setup if detected
if [[ $HAS_PYTHON_REQS -eq 1 || $PYTHON_FEATURE -eq 1 ]]; then
    cat >> "$DOCKERFILE" << 'DOCKERFILE_EOF'

# Install Python and create pre-built virtual environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

DOCKERFILE_EOF

    if [[ $HAS_PYTHON_REQS -eq 1 ]]; then
        if [[ $HAS_PYTORCH -eq 1 ]]; then
            # PyTorch detected: install CPU-only version FIRST, then requirements,
            # then clean up GPU packages — all in one RUN layer to minimize image size.
            cat >> "$DOCKERFILE" << 'DOCKERFILE_EOF'
# Install CPU-only PyTorch FIRST to avoid pulling in GPU/triton packages,
# then install remaining requirements. Keep all installs and cleanups in
# one RUN layer so removed packages don't bloat earlier layers.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/py_env && \
    /opt/py_env/bin/pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu && \
    /opt/py_env/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    /opt/py_env/bin/pip uninstall -y \
        triton nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
        nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
        nvidia-cufft-cu12 nvidia-cufile-cu12 \
        nvidia-curand-cu12 nvidia-cusparse-cu12 \
        nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
        nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 \
        nvidia-nvtx-cu12 2>/dev/null || true && \
    find /opt/py_env -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/py_env -name "*.pyc" -delete 2>/dev/null || true
DOCKERFILE_EOF
        else
            cat >> "$DOCKERFILE" << 'DOCKERFILE_EOF'
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/py_env && \
    /opt/py_env/bin/pip install --no-cache-dir -r /tmp/requirements.txt
DOCKERFILE_EOF
        fi
    elif [[ $HAS_PYPROJECT -eq 1 ]]; then
        cat >> "$DOCKERFILE" << 'DOCKERFILE_EOF'
COPY pyproject.toml /tmp/pyproject.toml
RUN python3 -m venv /opt/py_env && \
    /opt/py_env/bin/pip install --no-cache-dir /tmp/
DOCKERFILE_EOF
    fi
fi

# Add Node.js setup if detected
if [[ $HAS_PACKAGE_JSON -eq 1 || $NODE_FEATURE -eq 1 ]]; then
    cat >> "$DOCKERFILE" << 'DOCKERFILE_EOF'

# Install Node.js LTS and pre-install dependencies
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

DOCKERFILE_EOF

    if [[ $HAS_PACKAGE_JSON -eq 1 ]]; then
        cat >> "$DOCKERFILE" << 'DOCKERFILE_EOF'
COPY package.json package-lock.json* /tmp/app/
RUN cd /tmp/app && npm ci --ignore-scripts 2>/dev/null || npm install --ignore-scripts
DOCKERFILE_EOF
    fi
fi

echo "  Generated: $DOCKERFILE"

# ── Build the rebuild triggers table for docs ─────────────────────────
REBUILD_TRIGGERS="| File changed | Why rebuild is needed |\n|---|---|"
REBUILD_TRIGGERS+="\n| \`.devcontainer/Dockerfile\` | Image build instructions changed |"
[[ $HAS_PYTHON_REQS -eq 1 ]] && REBUILD_TRIGGERS+="\n| \`requirements.txt\` | Python packages are pre-installed in \`/opt/py_env\` |"
[[ $HAS_PYPROJECT -eq 1 ]]   && REBUILD_TRIGGERS+="\n| \`pyproject.toml\` | Python packages are pre-installed in \`/opt/py_env\` |"
[[ $HAS_PACKAGE_JSON -eq 1 ]] && REBUILD_TRIGGERS+="\n| \`package.json\` / \`package-lock.json\` | Node modules are pre-installed |"

# ── Determine org vs user package URL ─────────────────────────────────
# Check if the GHCR org looks like a GitHub org (has a repo with multiple members)
# Default to org path; users can adjust in the generated doc
GHCR_SETTINGS_URL="github.com/orgs/$GHCR_ORG/packages/container/$IMAGE_NAME/settings"

# ── Generate IMAGE-MAINTENANCE.md ────────────────────────────────────
MAINT_DOC="$DEVCONTAINER_DIR/IMAGE-MAINTENANCE.md"
echo "Generating .devcontainer/IMAGE-MAINTENANCE.md..."

cat > "$MAINT_DOC" << MAINTEOF
# Dev Container Image Maintenance

The Codespaces dev container uses a pre-built Docker image hosted on GitHub Container Registry (GHCR) to avoid large downloads at startup.

## Image location

\`\`\`
$FULL_IMAGE
\`\`\`

## When to rebuild the image

Rebuild whenever you change any of these files:

$(echo -e "$REBUILD_TRIGGERS")

Changes to these files do **not** require a rebuild:

- \`.devcontainer/devcontainer.json\` (VS Code settings, extensions, lifecycle commands)
- Application source code, data files, or docs
- Scripts that run at startup (referenced in postCreateCommand)

## How to rebuild and push

Run from the repo root.

**You must specify \`--platform linux/amd64\`** — Codespaces runs on amd64. If you build on Apple Silicon (ARM) without this flag, the image will be arm64 and Codespaces will fail with "No manifest found."

\`\`\`bash
# 1. Log in to GHCR (one-time, or when token expires)
docker login ghcr.io -u YOUR_GITHUB_USERNAME

# 2. Build the image (always specify platform)
docker build --platform linux/amd64 -f .devcontainer/Dockerfile -t $FULL_IMAGE .

# 3. Push to GHCR
docker push $FULL_IMAGE
\`\`\`

Then commit and push your code changes so the repo and image stay in sync.

## Image size constraints

Free Codespaces have a 32GB storage limit. The image must fit within this along with Codespaces system overhead (agent, VS Code server, mounts). To keep the image small:
MAINTEOF

if [[ $HAS_PYTORCH -eq 1 ]]; then
    cat >> "$MAINT_DOC" << 'MAINTEOF'

- **Install CPU-only PyTorch first** (before `requirements.txt`) using `--index-url https://download.pytorch.org/whl/cpu`. This prevents pip from ever downloading the GPU variant.
- **Remove `triton`** — a ~1GB PyTorch dependency not needed for CPU inference. Include it in the `pip uninstall` list in the Dockerfile.
- **Remove all `nvidia-*` packages** — CUDA libraries are not usable in Codespaces.
MAINTEOF
fi

cat >> "$MAINT_DOC" << 'MAINTEOF'

- **Clean up test directories and .pyc files** to recover additional space.
- **Keep all installs and removals in the same `RUN` layer** — otherwise Docker preserves deleted files in earlier layers and the image stays large.

If a new dependency causes the image to exceed 32GB, check for unnecessary transitive dependencies (`pip show <package>`) and uninstall them in the Dockerfile.

## GHCR package visibility

Codespaces must be able to pull the image. There are two options:

### Option A: Make the package public

MAINTEOF

cat >> "$MAINT_DOC" << MAINTEOF
Go to: \`$GHCR_SETTINGS_URL\`
MAINTEOF

cat >> "$MAINT_DOC" << 'MAINTEOF'

Under "Danger Zone", set visibility to **Public**. This may require org admin permissions.

### Option B: Grant repository access (for private packages)

If org restrictions prevent making the package public:

1. Go to the package settings page (same URL as above)
2. Under **"Manage Actions access"**, add the repository with **Read** access
3. Under **"Manage repository access"**, also add the repository with **Read** access

Both access grants are needed for Codespaces to authenticate the pull.

## Startup script considerations

If your `postCreateCommand` runs scripts that install packages (e.g., `pip install`), those commands must be **conditional** — they should only run when the pre-built venv is NOT available. Otherwise the script will re-download packages that are already baked into the image, defeating the purpose of pre-building.

Example pattern for a setup script:

```bash
if [ -d "/opt/py_env" ]; then
    # Fast path: copy pre-built venv from image
    cp -a /opt/py_env ./py_env
    # Fix hardcoded paths
    sed -i "s|/opt/py_env|$(pwd)/py_env|g" ./py_env/bin/activate
    sed -i "s|/opt/py_env|$(pwd)/py_env|g" ./py_env/bin/pip*
    sed -i "s|/opt/py_env|$(pwd)/py_env|g" ./py_env/pyvenv.cfg 2>/dev/null || true
else
    # Fallback: create venv and install from scratch
    python3 -m venv ./py_env
    pip install -r requirements.txt
    # Any pip install --upgrade commands go HERE, not outside the if/else
fi
```

Any `pip install`, `pip install --upgrade`, or NVIDIA/CUDA cleanup commands must be inside the `else` block. If they run unconditionally after the copy, they will download new versions over the network on every codespace creation.

## Forked repos and prebuilds

Forked repos **cannot pull GHCR images** from the original org's private packages. Even if the image is public, forks should use the Dockerfile build approach for reliability.

### devcontainer.json for forks

Forks must use `"build"` instead of `"image"`, and **must include `"context": ".."`** because Codespaces uses `.devcontainer/` as the default build context, but the Dockerfile needs to `COPY` files (e.g. `requirements.txt`) from the repo root:

```jsonc
// For the original repo (pulls pre-built image — fast):
"image": "ghcr.io/ORG/IMAGE:latest",

// For forks or when GHCR image is unavailable (builds from Dockerfile):
"build": {
    "dockerfile": "Dockerfile",
    "context": ".."
},
```

Without `"context": ".."`, the build will fail with:
```
"/requirements.txt": not found
```

### Setting up prebuilds on a fork

1. In the forked repo, change `devcontainer.json` to use the `"build"` config shown above
2. Go to the fork's **Settings > Codespaces > Prebuilds > Set up prebuild**
3. Configure for the `main` branch and your preferred region
4. The prebuild will build the Dockerfile on GitHub's infra and cache the result
5. Subsequent codespace creations from the fork will use the cached prebuild

This is slower than pulling a pre-built image (~5-10 min for the initial prebuild) but only happens once. After the prebuild completes, users get fast startup.

### Keeping forks in sync

When the upstream repo updates dependency files or the Dockerfile:

1. Sync the fork with upstream
2. The prebuild will automatically re-trigger (if configured for "on push")
3. No need to rebuild or push any GHCR image — the fork builds its own

## Fallback: building without a pre-built image

If the GHCR image is unavailable, change `devcontainer.json` to build locally using the same fork-compatible config:

```jsonc
"build": {
    "dockerfile": "Dockerfile",
    "context": ".."
},
```

With local build, Codespaces builds the Dockerfile during creation on GitHub's infrastructure (not on user WiFi). This is slower (~5-10 min) but works without any GHCR setup.

## Testing after a rebuild

1. Create a new codespace from the main branch
2. Confirm dependencies are already installed (no downloads during startup)
3. Confirm the startup script does NOT run pip install (check the creation log)
4. Run your application to verify everything works
5. Check disk usage with `df -h` — ensure adequate free space remains
MAINTEOF

echo "  Generated: $MAINT_DOC"

# ── Update devcontainer.json ──────────────────────────────────────────
echo ""
echo "Updating .devcontainer/devcontainer.json..."

cp "$DEVCONTAINER_JSON" "${DEVCONTAINER_JSON}${BACKUP_SUFFIX}"
echo "  Backed up original to devcontainer.json${BACKUP_SUFFIX}"

python3 << PYEOF
import json, re

with open("$DEVCONTAINER_JSON") as f:
    text = f.read()

# Strip JS-style comments for parsing
clean = re.sub(r'//.*', '', text)
data = json.loads(clean)

# Switch to pre-built image
if "build" in data:
    del data["build"]
data["image"] = "$FULL_IMAGE"

# Also store the fork-compatible build config as a comment hint
# (users switch to this when forking — see IMAGE-MAINTENANCE.md)
data["_comment_fork_build"] = "For forks, replace 'image' with: \"build\": {\"dockerfile\": \"Dockerfile\", \"context\": \"..\"}"

# Remove features that are now in the Dockerfile
features_to_remove = []
for key in data.get("features", {}):
    key_lower = key.lower()
    if "python" in key_lower and $HAS_PYTHON_REQS:
        features_to_remove.append(key)
    if "node" in key_lower and ($HAS_PACKAGE_JSON or $NODE_FEATURE):
        features_to_remove.append(key)

for key in features_to_remove:
    del data["features"][key]

# Remove empty features block
if "features" in data and not data["features"]:
    del data["features"]

with open("$DEVCONTAINER_JSON", "w") as f:
    json.dump(data, f, indent=4)
    f.write("\n")

if features_to_remove:
    print("  Removed features (now in Dockerfile): " + ", ".join(features_to_remove))
print("  Set image to: $FULL_IMAGE")
print("  Added fork build hint (see _comment_fork_build in devcontainer.json)")
PYEOF

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Done! Files created/updated:"
echo "============================================================"
echo "  .devcontainer/Dockerfile           (generated)"
echo "  .devcontainer/IMAGE-MAINTENANCE.md (generated)"
echo "  .devcontainer/devcontainer.json    (updated)"
echo ""
echo "  Next steps:"
echo "  ────────────────────────────────────────────────"
echo "  1. Review the generated Dockerfile and customize"
echo "     (add project-specific tools, pre-download models, etc.)"
echo ""
echo "  2. Build and push the image:"
echo ""
echo "     docker login ghcr.io -u YOUR_GITHUB_USERNAME"
echo "     docker build --platform linux/amd64 -f .devcontainer/Dockerfile -t $FULL_IMAGE ."
echo "     docker push $FULL_IMAGE"
echo ""
echo "  3. Make the GHCR package accessible (see IMAGE-MAINTENANCE.md):"
echo "     $GHCR_SETTINGS_URL"
echo ""
echo "  4. Check postCreateCommand scripts for unconditional pip installs"
echo "     (see IMAGE-MAINTENANCE.md 'Startup script considerations')"
echo ""
echo "  5. Commit and push all changes"
echo ""
echo "  6. (Optional) Enable Codespaces prebuilds:"
echo "     Repo Settings > Codespaces > Prebuilds > Set up prebuild"
echo ""
echo "  For FORKED repos:"
echo "  ────────────────────────────────────────────────"
echo "  Forks cannot pull the GHCR image. In the fork's"
echo "  devcontainer.json, replace the \"image\" line with:"
echo ""
echo '     "build": {'
echo '         "dockerfile": "Dockerfile",'
echo '         "context": ".."'
echo '     },'
echo ""
echo "  The \"context\": \"..\" is required because Codespaces"
echo "  uses .devcontainer/ as the build context by default,"
echo "  but the Dockerfile COPYs files from the repo root."
echo "  See IMAGE-MAINTENANCE.md for full details."
echo "============================================================"
