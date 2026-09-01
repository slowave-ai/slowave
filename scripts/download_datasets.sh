#!/usr/bin/env bash
# Download benchmark datasets required for full reproducibility.
#
# Datasets are NOT included in the repository:
#  - LongMemEval files are large (15–265 MB) and subject to their own license
#    (https://github.com/xiaowu0162/LongMemEval — check before redistributing)
#  - LoCoMo is a public ACL 2024 dataset (2.7 MB)
#  - DMR (MSC-Self-Instruct) is Apache-2.0 from HuggingFace/MemGPT (8.2 MB)
#  - StaleMemory is a synthetically generated benchmark included here by value
#    but withheld from git due to size (13 MB)
#
# Usage:
#   bash scripts/download_datasets.sh            # all datasets
#   bash scripts/download_datasets.sh locomo     # only LoCoMo
#   bash scripts/download_datasets.sh lme        # only LongMemEval

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TARGET="${1:-all}"

# ── LoCoMo ──────────────────────────────────────────────────────────────────
download_locomo() {
    echo "→ LoCoMo (ACL 2024, ~2.7 MB) ..."
    mkdir -p "$REPO_ROOT/data/locomo"
    curl -fL --progress-bar \
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json" \
        -o "$REPO_ROOT/data/locomo/locomo10.json"
    echo "  ✓ data/locomo/locomo10.json"
}

# ── LongMemEval ─────────────────────────────────────────────────────────────
download_longmemeval() {
    echo "→ LongMemEval oracle split (~15 MB) ..."
    mkdir -p "$REPO_ROOT/data/longmemeval"
    curl -fL --progress-bar \
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json" \
        -o "$REPO_ROOT/data/longmemeval/longmemeval_oracle.json"
    echo "  ✓ data/longmemeval/longmemeval_oracle.json"

    echo "→ LongMemEval full haystack (~265 MB, needed for 93.4% haystack run) ..."
    curl -fL --progress-bar \
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" \
        -o "$REPO_ROOT/data/longmemeval/longmemeval_s_cleaned.json"
    echo "  ✓ data/longmemeval/longmemeval_s_cleaned.json"
}

# ── DMR (MSC-Self-Instruct, Apache-2.0) ──────────────────────────────────────
download_dmr() {
    echo "→ DMR / MSC-Self-Instruct (Apache-2.0, ~8.2 MB) ..."
    mkdir -p "$REPO_ROOT/data/dmr_original"
    curl -fL --progress-bar \
        "https://huggingface.co/datasets/MemGPT/MSC-Self-Instruct/resolve/main/msc_self_instruct.jsonl" \
        -o "$REPO_ROOT/data/dmr_original/msc_self_instruct.jsonl"
    echo "  ✓ data/dmr_original/msc_self_instruct.jsonl"
}

# ── StaleMemory (synthetically generated) ────────────────────────────────────
download_stalememory() {
    echo "→ StaleMemory (~13 MB, synthetically generated benchmark) ..."
    mkdir -p "$REPO_ROOT/data/stalememory"
    # Hosted in the Slowave releases as a standalone asset. There is no local
    # generator in this checkout, so fail clearly when the asset is unavailable.
    if ! curl -fL --progress-bar \
        "https://github.com/mrsalty/slowave/releases/download/datasets/stalememory_scenarios.tar.gz" \
        -o /tmp/stalememory_scenarios.tar.gz 2>/dev/null; then
        echo "  ✗ StaleMemory release asset not available; obtain scenarios.jsonl from the project maintainer."
        return 1
    fi
    tar -xzf /tmp/stalememory_scenarios.tar.gz -C "$REPO_ROOT/data/stalememory/"
    rm /tmp/stalememory_scenarios.tar.gz
    echo "  ✓ data/stalememory/scenarios.jsonl + manifest.json"
}

case "$TARGET" in
    locomo)     download_locomo ;;
    lme)        download_longmemeval ;;
    dmr)        download_dmr ;;
    stalememory) download_stalememory ;;
    all)
        download_locomo
        download_longmemeval
        download_dmr
        download_stalememory
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: $0 [all|locomo|lme|dmr|stalememory]"
        exit 1
        ;;
esac

echo ""
echo "Done. Run the benchmarks:"
echo "  python tests/integration/run_full_benchmark.py"
