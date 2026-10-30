#!/usr/bin/env bash
# Expose ~/models/llama2-7b-hf (the sha256-checked copy) as meta-llama/Llama-2-7b-hf in the HF cache,
# so Prismatic's from_pretrained("meta-llama/Llama-2-7b-hf") resolves it with HF_HUB_OFFLINE=1.
set -euo pipefail
REV=$1
SRC=$HOME/models/llama2-7b-hf
C=${HF_HOME:-$HOME/.cache/huggingface}/hub/models--meta-llama--Llama-2-7b-hf
mkdir -p $C/refs $C/snapshots/$REV
printf "%s" $REV > $C/refs/main
for f in config.json generation_config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
         model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json \
         LICENSE.txt USE_POLICY.md; do
  ln -sfn $SRC/$f $C/snapshots/$REV/$f
done
ls -la $C/snapshots/$REV | head -14
