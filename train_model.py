"""
train_model.py
--------------
End-to-end training script.
Run with:
    python train_model.py
"""

import sys
import joblib

from pipeline.data_ingestion import load_data
from pipeline.data_validation import validate_data
from pipeline.data_preprocessing import preprocess_data
from pipeline.model_trainer import train_and_evaluate

MODEL_OUTPUT_PATH = "model.pkl"
DATA_PATH = "data/raw.csv"


def main() -> None:
    print("=" * 60)
    print("  Heart Disease ML Pipeline — Training Run")
    print("=" * 60)

    # Step 1 — Ingest
    df_raw = load_data(DATA_PATH)

    # Step 2 — Validate
    df_validated = validate_data(df_raw)

    # Step 3 — Preprocess
    X, y = preprocess_data(df_validated)

    # Step 4 — Train & evaluate
    best_model, results = train_and_evaluate(X, y)

    # Step 5 — Save best model
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"\n[Saving] Best model saved to '{MODEL_OUTPUT_PATH}'.")

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    for key in ("baseline", "improved"):
        r = results[key]
        m = r["metrics"]
        print(
            f"  {r['name']:<30}  "
            f"Acc={m['accuracy']:.4f}  "
            f"P={m['precision']:.4f}  "
            f"R={m['recall']:.4f}  "
            f"F1={m['f1_score']:.4f}"
        )
    best = results["best"]
    print(f"\n  ★  Best model: {best['name']} (F1={best['metrics']['f1_score']:.4f})")
    print("=" * 60)
    print("\n✅  Pipeline complete. Run the API with:")
    print("      uvicorn api.api:app --reload\n")


if __name__ == "__main__":
    main()
