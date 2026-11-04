#!/usr/bin/env bash
# Records the server's real hardware and installed models in docs/system-info.txt.
# Read-only. Leaves out serial numbers, hostnames, IP addresses and full file paths.
# Review the output file before committing it.
#   bash scripts/collect_system_info.sh
set -u
out=docs/system-info.txt
{
  echo "# Collected $(date +%F)"
  echo; echo "## CPU"; lscpu | grep -E "^(Model name|Socket|Core|Thread)"
  echo; echo "## Memory"; free -h | awk 'NR<=2'
  echo; echo "## GPUs"
  nvidia-smi --query-gpu=name,memory.total,power.limit,driver_version --format=csv
  nvidia-smi | grep -o "CUDA Version: [0-9.]*"
  echo; echo "## Drives"; lsblk -d -o MODEL,SIZE,ROTA,TRAN
  echo; echo "## OS"; lsb_release -ds; uname -r
  echo; echo "## Model files (names and sizes only)"
  find / -type f -iname "*.gguf" -not -path "/proc/*" -not -path "/sys/*" -printf "%f  %s bytes\n" 2>/dev/null | sort
} > "$out"
echo "wrote $out - read it before committing"
