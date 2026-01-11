# export_scaler.py
import joblib
import gzip

# Path to your compressed scaler file
input_path = "scaler.gz"
output_path = "models/scaler.pkl"

# Load from gzipped file
with gzip.open(input_path, "rb") as f:
    scaler = joblib.load(f)

# Save properly as .pkl
joblib.dump(scaler, output_path)
print(f"Scaler exported successfully to {output_path}")
