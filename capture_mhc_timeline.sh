#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./capture_mhc_timeline.sh [full|smoke] [options]

Modes:
  full                 Capture the default full benchmark timeline.
  smoke                Capture a short one-rep timeline.

Shape options:
  --shape S,B,H        Run one shape: seqlen S, batch B, hidden H.
  --seqlens LIST       Comma-separated seqlens, e.g. 4096,8192.
  --batches LIST       Comma-separated batch sizes, e.g. 1,4.
  --hiddens LIST       Comma-separated hidden dims, e.g. 4096,7168.

Other options:
  --out-dir DIR        Output directory. Default: $PROJECT_ROOT/../mHC-results.
  --mhc-bench-path DIR Use an external mhc_bench checkout instead of vendored kernels.
  --trace TRACE        Nsight trace set. Default: cuda,nvtx.
  --pytest-bin BIN     Pytest executable. Default: pytest.
  -h, --help           Show this help.

Environment overrides:
  PROJECT_ROOT         Project root. Default: directory containing this script.
  OUT_DIR              Same as --out-dir.
  MHC_BENCH_PATH       Same as --mhc-bench-path.
  TRACE                Same as --trace.
  PYTEST_BIN           Same as --pytest-bin.
USAGE
}

MODE="full"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEQLENS="${MHC_BENCH_SEQLENS:-}"
BATCHES="${MHC_BENCH_BATCH_SIZES:-}"
HIDDENS="${MHC_BENCH_HIDDENS:-}"
OUT_DIR_ARG="${OUT_DIR:-}"
MHC_BENCH_PATH_ARG="${MHC_BENCH_PATH:-}"
TRACE="${TRACE:-cuda,nvtx}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    full|smoke)
      MODE="$1"
      shift
      ;;
    --shape)
      if [[ $# -lt 2 ]]; then
        echo "--shape requires S,B,H" >&2
        exit 2
      fi
      shape="${2//x/,}"
      IFS=',' read -r shape_s shape_b shape_h shape_extra <<< "${shape}"
      if [[ -z "${shape_s:-}" || -z "${shape_b:-}" || -z "${shape_h:-}" || -n "${shape_extra:-}" ]]; then
        echo "--shape must be S,B,H, for example --shape 4096,1,7168" >&2
        exit 2
      fi
      SEQLENS="${shape_s}"
      BATCHES="${shape_b}"
      HIDDENS="${shape_h}"
      shift 2
      ;;
    --seqlens)
      if [[ $# -lt 2 ]]; then
        echo "--seqlens requires a comma-separated list" >&2
        exit 2
      fi
      SEQLENS="$2"
      shift 2
      ;;
    --batches|--batch-sizes)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a comma-separated list" >&2
        exit 2
      fi
      BATCHES="$2"
      shift 2
      ;;
    --hiddens|--hidden-dims)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a comma-separated list" >&2
        exit 2
      fi
      HIDDENS="$2"
      shift 2
      ;;
    --out-dir)
      if [[ $# -lt 2 ]]; then
        echo "--out-dir requires a directory" >&2
        exit 2
      fi
      OUT_DIR_ARG="$2"
      shift 2
      ;;
    --mhc-bench-path)
      if [[ $# -lt 2 ]]; then
        echo "--mhc-bench-path requires a directory" >&2
        exit 2
      fi
      MHC_BENCH_PATH_ARG="$2"
      shift 2
      ;;
    --trace)
      if [[ $# -lt 2 ]]; then
        echo "--trace requires a trace set" >&2
        exit 2
      fi
      TRACE="$2"
      shift 2
      ;;
    --pytest-bin)
      if [[ $# -lt 2 ]]; then
        echo "--pytest-bin requires an executable" >&2
        exit 2
      fi
      PYTEST_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
OUT_DIR="${OUT_DIR_ARG:-${PROJECT_ROOT}/../mHC-results}"

mkdir -p "${OUT_DIR}"
cd "${PROJECT_ROOT}"

if [[ -n "${SEQLENS}" ]]; then
  export MHC_BENCH_SEQLENS="${SEQLENS}"
fi
if [[ -n "${BATCHES}" ]]; then
  export MHC_BENCH_BATCH_SIZES="${BATCHES}"
fi
if [[ -n "${HIDDENS}" ]]; then
  export MHC_BENCH_HIDDENS="${HIDDENS}"
fi
if [[ -n "${MHC_BENCH_PATH_ARG}" ]]; then
  export MHC_BENCH_PATH="${MHC_BENCH_PATH_ARG}"
fi

case "${MODE}" in
  full)
    OUT_PREFIX="${OUT_DIR}/mhc-three-backends"
    EXTRA_PYTEST_ARGS=()
    ;;
  smoke)
    OUT_PREFIX="${OUT_DIR}/mhc-three-backends-smoke"
    EXTRA_PYTEST_ARGS=(--tk-bench-warmup 1 --tk-bench-rep 1)
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

echo "Project root: ${PROJECT_ROOT}"
echo "Output prefix: ${OUT_PREFIX}"
echo "Trace: ${TRACE}"
echo "Shapes: seqlens=${MHC_BENCH_SEQLENS:-default}, batches=${MHC_BENCH_BATCH_SIZES:-default}, hiddens=${MHC_BENCH_HIDDENS:-default}"
echo "MHC_BENCH_PATH: ${MHC_BENCH_PATH:-vendored}"

nsys profile --sample=none --cpuctxsw=none -t "${TRACE}" \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node -f true -x true \
  -o "${OUT_PREFIX}" \
  "${PYTEST_BIN}" tests/mhc/test_megatron_mhc_benchmark.py \
    --run-benchmark -m benchmark --nsys-capture \
    "${EXTRA_PYTEST_ARGS[@]}" \
    --benchmark-output "${OUT_PREFIX}.jsonl"

nsys stats --report cuda_gpu_kern_sum "${OUT_PREFIX}.nsys-rep"

echo
echo "Timeline: ${OUT_PREFIX}.nsys-rep"
echo "JSONL:    ${OUT_PREFIX}.jsonl"
