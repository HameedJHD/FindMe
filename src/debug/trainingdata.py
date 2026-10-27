import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"
PETA_DIR = MODELS_DIR / "peta"
REID_DIR = MODELS_DIR / "reid"


def print_section(title):
    print(f"=== {title} ===")


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def inspect_json_models(models_dir):
    model_files = sorted(models_dir.rglob("*.json"))
    if not model_files:
        print(f"No JSON metadata found under {models_dir}")
        return

    print_section(f"JSON metadata in {models_dir}")
    print(f"Found {len(model_files)} file(s)\n")

    for model_file in model_files:
        try:
            with model_file.open("r", encoding="utf-8") as handle:
                model_data = json.load(handle)

            print(f"[{model_file.relative_to(models_dir)}]")
            if isinstance(model_data, dict):
                for key, value in model_data.items():
                    print(f"{key}: {value}")
            else:
                print(model_data)
            print()
        except Exception as exc:
            print(f"Error reading {model_file}: {exc}\n")


def _read_yaml_summary(args_path):
    if not args_path.exists():
        return None

    try:
        with args_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:
        return {"error": str(exc)}

    if isinstance(data, dict):
        keys = ["task", "mode", "model", "data", "epochs", "batch", "imgsz", "device", "name"]
        return {key: data.get(key) for key in keys if key in data}

    return {"value": data}


def _pick_best_row(rows, primary_metric):
    best_row = None
    best_value = None
    for row in rows:
        value = _to_float(row.get(primary_metric))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_row = row
            best_value = value
    return best_row, best_value


def inspect_yolo_runs(runs_dir):
    results_files = sorted(runs_dir.rglob("results.csv"))
    if not results_files:
        print(f"No YOLO results.csv files found under {runs_dir}\n")
        return

    print_section(f"YOLO run metrics in {runs_dir}")
    for results_file in results_files:
        try:
            with results_file.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            if not rows:
                print(f"[{results_file.relative_to(ROOT)}]")
                print("No metric rows found\n")
                continue

            primary_metric = next(
                (
                    name
                    for name in (
                        "metrics/mAP50-95(B)",
                        "metrics/mAP50(B)",
                        "metrics/precision(B)",
                        "metrics/recall(B)",
                    )
                    if name in rows[0]
                ),
                None,
            )
            final_row = rows[-1]
            best_row, best_value = (
                _pick_best_row(rows, primary_metric) if primary_metric else (None, None)
            )

            args_summary = _read_yaml_summary(results_file.with_name("args.yaml"))

            print(f"[{results_file.relative_to(ROOT)}]")
            if args_summary:
                print("run args:")
                for key, value in args_summary.items():
                    print(f"  {key}: {value}")

            print(f"epochs logged: {len(rows)}")
            if primary_metric:
                print(f"primary metric: {primary_metric}")
                print(f"final {primary_metric}: {final_row.get(primary_metric)}")
                if best_row is not None:
                    print(f"best {primary_metric}: {best_value} at epoch {best_row.get('epoch')}")

            print("final row:")
            for key in (
                "epoch",
                "train/box_loss",
                "train/cls_loss",
                "train/dfl_loss",
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "val/box_loss",
                "val/cls_loss",
                "val/dfl_loss",
            ):
                if key in final_row:
                    print(f"  {key}: {final_row.get(key)}")
            print()
        except Exception as exc:
            print(f"Error reading {results_file}: {exc}\n")


def inspect_model_folders():
    print_section("Other stored model artifacts")

    for label, folder, patterns in (
        ("PETA", PETA_DIR, ("*.pt", "*.pth", "*.json", "*.csv")),
        ("Re-ID", REID_DIR, ("*.pt", "*.pth", "*.json", "*.csv")),
    ):
        if not folder.exists():
            print(f"{label}: folder not found at {folder}")
            continue

        files = []
        for pattern in patterns:
            files.extend(sorted(folder.rglob(pattern)))

        if not files:
            print(f"{label}: no stored artifacts found")
            continue

        print(f"{label}: found {len(files)} artifact(s)")
        for file_path in files[:5]:
            print(f"  {file_path.relative_to(ROOT)}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")

        print("  no persisted evaluation metrics were found in these artifacts")
        print()


def main():
    if MODELS_DIR.exists():
        inspect_json_models(MODELS_DIR)
    else:
        print(f"Models directory not found: {MODELS_DIR}")

    print()

    if RUNS_DIR.exists():
        inspect_yolo_runs(RUNS_DIR)
    else:
        print(f"Runs directory not found: {RUNS_DIR}")

    print()
    inspect_model_folders()


if __name__ == "__main__":
    main()
