#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
usage() {
    cat <<'EOF'
Usage: bash rotor.sh COMMAND [OPTIONS]
  setup       Create .venv and install requirements (internet required)
  check       Check Python packages and CUDA availability
  extract     Run extract_nd.py
  visualize   Run visualize_cylindrical_data.py (save PNG)
  view3d      Convert prediction NPZ to interactive HTML (--input required)
  train       Run train_nd.py (joint ur/utheta/uz)
  evaluate    Run evaluate_error.py (validation cases)
  predict     Open interactive prediction plot with mouse velocity readout; save NPZ
  test        Run synthetic OpenFOAM pipeline test
  help        Show this help
Example: bash rotor.sh train --epochs 100 --batch-size 4 --lambda-gradient <weight>
Example: bash rotor.sh extract --data-root /data/OpenFOAM
Prediction: bash rotor.sh predict --rd 9.3101751 --l-over-d 1.6 --ground-z <actual> --z-max <actual>
Command options: bash rotor.sh train --help
Relative option paths are relative to your current terminal directory.
Defaults point to the directory containing rotor.sh.
EOF
}
command="${1:-help}"
if (( $# )); then shift; fi
case "$command" in
    help|-h|--help) usage; exit 0 ;;
    setup|check|extract|visualize|view3d|train|evaluate|predict|test) ;;
    *) echo "Unknown command: $command" >&2; usage >&2; exit 2 ;;
esac
if [[ "$command" == setup ]]; then
    if (( $# )); then echo 'setup accepts no options' >&2; exit 2; fi
    "${PYTHON:-python3}" -m venv "$ROOT/.venv"
    "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
    echo 'Setup complete. Run: bash rotor.sh check'
    exit 0
fi
PY="${ROTOR_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo 'Python environment missing. Run: bash rotor.sh setup' >&2
    echo 'Or set ROTOR_PYTHON to an existing Python executable absolute path.' >&2
    exit 1
fi
if [[ "$command" != predict ]]; then
    export MPLBACKEND=Agg
fi
export PYVISTA_OFF_SCREEN=true
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
case "$command" in
    check)
        "$PY" -c 'import sys, numpy, pandas, torch, matplotlib, pyvista; print("Python:", sys.version); print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"); print("All required packages imported successfully")'
        ;;
    extract)
        exec "$PY" "$ROOT/extract_nd.py" --csv "$ROOT/cases.csv" --output-root "$ROOT/dataset_zrd" "$@" ;;
    visualize)
        exec "$PY" "$ROOT/visualize_cylindrical_data.py" --data-root "$ROOT/dataset_zrd" --output-dir "$ROOT/results/visualize" "$@" ;;
    view3d)
        exec "$PY" "$ROOT/visualize_prediction_3d.py" "$@" ;;
    train)
        exec "$PY" "$ROOT/train_nd.py" --csv "$ROOT/cases.csv" --data-root "$ROOT/dataset_zrd" --model-dir "$ROOT/models_zrd" "$@" ;;
    evaluate)
        exec "$PY" "$ROOT/evaluate_error.py" --csv "$ROOT/cases.csv" --data-root "$ROOT/dataset_zrd" --model-dir "$ROOT/models_zrd" --output-dir "$ROOT/results/evaluate" --no-show "$@" ;;
    predict)
        exec "$PY" "$ROOT/predict_nd.py" --model-dir "$ROOT/models_zrd" --output "$ROOT/results/prediction_cylinder_r2RD.npz" "$@" ;;
    test)
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        exec "$PY" -m unittest discover -s "$ROOT" -p 'test_*.py' "$@" ;;
esac
