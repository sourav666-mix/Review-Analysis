import pickle
import joblib
import numpy as np
from pathlib import Path

models_dir = Path("models")
files = ["tfidf_vectorizer.pkl", "scaler.pkl", "linearsvc_tuned_model.pkl"]

print("=" * 60)
for f in files:
    path = models_dir / f
    print(f"\n📁 {f}")
    print(f"   Exists: {path.exists()}")
    if path.exists():
        print(f"   Size: {path.stat().st_size:,} bytes")
        with open(path, "rb") as file:
            header = file.read(8)
            print(f"   Header bytes: {header.hex()}")
        
        # Try pickle
        try:
            with open(path, "rb") as file:
                obj = pickle.load(file)
            print(f"   ✅ PICKLE OK: {type(obj).__name__}")
            continue
        except Exception as e:
            print(f"   ❌ Pickle: {e}")
        
        # Try joblib
        try:
            obj = joblib.load(path)
            print(f"   ✅ JOBLIB OK: {type(obj).__name__}")
        except Exception as e:
            print(f"   ❌ Joblib: {e}")