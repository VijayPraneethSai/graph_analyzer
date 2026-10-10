#!/usr/bin/env bash
# Runs CPU + GPU on every graph and appends to results/results.csv (or $RESULTS_CSV).
# Usage: scripts/run_bench.sh [--cpu-only] [--large] [graph ...]   (default: the three SNAP graphs)
#   --large  also runs data/com-LiveJournal.txt (fetch it with scripts/download_data.sh --large) when that
#            file exists; if it is missing it is skipped with a note and the default run is unaffected.
#            Explicitly listed graphs are still required to exist.
# Extra gpu_tc arguments via env: GPU_ARGS=--pinned scripts/run_bench.sh (pinned rows get a _pinned suffix).
# Densest-subgraph stage (separate from the triangle stage, which is unchanged): scripts/run_bench.sh --densest [graph ...]
#   runs ./bin/gpu_densest (or ./bin/cpu_densest with --cpu-only) instead of cpu_tc/gpu_tc and appends to
#   results/results_densest.csv (or $RESULTS_CSV). Extra program arguments via env, e.g.
#   DENSEST_ARGS="1 10 --no-greedy --verbose" scripts/run_bench.sh --densest data/as-Skitter.txt
#   (--no-greedy is advisable on as-Skitter / com-LiveJournal). Rows are appended only if the program exits 0.
# Stage-6 paper algorithms (GPU core decomposition + CPU CoreExact/Greedy++): scripts/run_bench.sh --dsg [graph ...]
#   runs ./bin/gpu_dsg (or ./bin/cpu_dsg with --cpu-only) and appends to results/results_dsg.csv (or $RESULTS_CSV). Extra program
#   arguments via env, e.g. DSG_ARGS="--T 10 --verify-full --pinned" scripts/run_bench.sh --dsg data/Wiki-Vote.txt
#   (add --no-greedy to DSG_ARGS on graphs with huge hubs). Rows are appended only if the program exits 0, i.e. only if the GPU
#   core numbers matched the CPU Alg 3 numbers element-wise (and --verify-full, if given, matched). Nothing is ever fabricated:
#   the CSV does not exist until a real run has produced it.
# Rows are appended only for programs that exited 0. A GPU run with a count MISMATCH
# (gpu_tc exits 1) never enters the results table.
cd "$(dirname "$0")/.." || exit 1
CPU_ONLY=0
LARGE=0
DENSEST=0
DSG=0
GRAPHS=()
for a in "$@"; do
  if [ "$a" = "--cpu-only" ]; then CPU_ONLY=1
  elif [ "$a" = "--large" ]; then LARGE=1
  elif [ "$a" = "--densest" ]; then DENSEST=1
  elif [ "$a" = "--dsg" ]; then DSG=1
  else GRAPHS+=("$a"); fi
done
[ ${#GRAPHS[@]} -eq 0 ] && GRAPHS=(data/Wiki-Vote.txt data/email-Enron.txt data/as-Skitter.txt)
if [ $LARGE -eq 1 ]; then
  if [ -s data/com-LiveJournal.txt ]; then GRAPHS+=(data/com-LiveJournal.txt)
  else echo "NOTE: --large given but data/com-LiveJournal.txt not found (run scripts/download_data.sh --large); skipping it"; fi
fi

# --- preflight: everything must exist before results.csv is touched
missing=0
for g in "${GRAPHS[@]}"; do
  if [ ! -s "$g" ]; then echo "ERROR: $g missing - run scripts/download_data.sh"; missing=1; fi
done
[ $missing -eq 1 ] && exit 1
if [ $DENSEST -eq 1 ]; then
  # ---- densest-subgraph stage: gpu_densest/cpu_densest print the CPU rows themselves (no separate cpu program) ----
  PROG=./bin/gpu_densest; [ $CPU_ONLY -eq 1 ] && PROG=./bin/cpu_densest
  if [ ! -x $PROG ]; then echo "ERROR: $PROG not built - run 'make cpu' (cpu_densest) or 'make ARCH=sm_75' on Colab (gpu_densest)"; exit 1; fi
  OUT=${RESULTS_CSV:-results/results_densest.csv}
  mkdir -p "$(dirname "$OUT")"
  [ -f "$OUT" ] || echo "impl,graph,n,m,best_triangles,best_vertices,density,rounds,time_ms,h2d_ms,total_ms,gpu,date" > "$OUT"
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  [ -z "$GPU" ] && GPU=none
  [ $CPU_ONLY -eq 1 ] && GPU=none
  DATE=$(date +%F)
  TMP=$(mktemp)
  trap 'rm -f "$TMP"' EXIT
  FAILED=()
  for g in "${GRAPHS[@]}"; do
    echo "== $g"
    $PROG "$g" $DENSEST_ARGS > "$TMP"; rc=$?
    cat "$TMP"
    if [ $rc -ne 0 ]; then
      echo "!!!!!! ERROR: $PROG exited $rc on $g (verification MISMATCH or CUDA error) - rows NOT appended to $OUT !!!!!!"
      FAILED+=("$PROG on $g (exit $rc, rows discarded)")
    else
      while IFS= read -r l; do echo "$l,\"$GPU\",$DATE" >> "$OUT"; done < "$TMP"
    fi
  done
  if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED:"; printf '  %s\n' "${FAILED[@]}"; exit 1; fi
  echo "appended to $OUT"
  exit 0
fi
if [ $DSG -eq 1 ]; then
  # ---- stage 6: gpu_dsg/cpu_dsg print all rows themselves ----
  PROG=./bin/gpu_dsg; [ $CPU_ONLY -eq 1 ] && PROG=./bin/cpu_dsg
  if [ ! -x $PROG ]; then echo "ERROR: $PROG not built - run 'make cpu' (cpu_dsg) or 'make ARCH=sm_75' on Colab (gpu_dsg)"; exit 1; fi
  OUT=${RESULTS_CSV:-results/results_dsg.csv}
  mkdir -p "$(dirname "$OUT")"
  [ -f "$OUT" ] || echo "impl,graph,n,m,triangles,vertices,density,time_ms,h2d_ms,total_ms,rounds,kmax,gpu,date" > "$OUT"
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  [ -z "$GPU" ] && GPU=none
  [ $CPU_ONLY -eq 1 ] && GPU=none
  DATE=$(date +%F)
  TMP=$(mktemp)
  trap 'rm -f "$TMP"' EXIT
  FAILED=()
  for g in "${GRAPHS[@]}"; do
    echo "== $g"
    $PROG "$g" $DSG_ARGS > "$TMP"; rc=$?
    cat "$TMP"
    if [ $rc -ne 0 ]; then
      echo "!!!!!! ERROR: $PROG exited $rc on $g (verification MISMATCH or CUDA error) - rows NOT appended to $OUT !!!!!!"
      FAILED+=("$PROG on $g (exit $rc, rows discarded)")
    else
      while IFS= read -r l; do echo "$l,\"$GPU\",$DATE" >> "$OUT"; done < "$TMP"
    fi
  done
  if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED:"; printf '  %s\n' "${FAILED[@]}"; exit 1; fi
  echo "appended to $OUT"
  exit 0
fi
if [ ! -x ./bin/cpu_tc ]; then echo "ERROR: ./bin/cpu_tc not built - run 'make cpu'"; exit 1; fi
if [ $CPU_ONLY -eq 0 ] && [ ! -x ./bin/gpu_tc ]; then
  echo "ERROR: ./bin/gpu_tc not built (no GPU build here?) - run 'make ARCH=sm_75' on Colab, or pass --cpu-only"
  exit 1
fi

OUT=${RESULTS_CSV:-results/results.csv}
mkdir -p "$(dirname "$OUT")"
[ -f "$OUT" ] || echo "impl,graph,n,m,triangles,time_ms,h2d_ms,gpu_total_ms,gpu,date" > "$OUT"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
[ -z "$GPU" ] && GPU=none
DATE=$(date +%F)
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
FAILED=()

for g in "${GRAPHS[@]}"; do
  echo "== $g"
  ./bin/cpu_tc "$g" 3 > "$TMP"; rc=$?
  cat "$TMP"
  if [ $rc -ne 0 ]; then
    echo "ERROR: cpu_tc failed (exit $rc) on $g - not appended"
    FAILED+=("cpu_tc on $g (exit $rc)")
  else
    while IFS= read -r l; do echo "$l,,,\"$GPU\",$DATE" >> "$OUT"; done < "$TMP"
  fi
  [ $CPU_ONLY -eq 1 ] && continue
  ./bin/gpu_tc "$g" 5 $GPU_ARGS > "$TMP"; rc=$?
  cat "$TMP"
  if [ $rc -ne 0 ]; then
    echo "!!!!!! ERROR: gpu_tc exited $rc on $g (count MISMATCH or CUDA error) - GPU rows NOT appended to $OUT !!!!!!"
    FAILED+=("gpu_tc on $g (exit $rc, rows discarded)")
  else
    # gpu line: impl,graph,n,m,tri,kernel_ms,h2d_ms,total_ms
    while IFS= read -r l; do echo "$l,\"$GPU\",$DATE" >> "$OUT"; done < "$TMP"
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED:"; printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "appended to $OUT"
