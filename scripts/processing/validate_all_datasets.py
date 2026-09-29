from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
OUT = P / "dataset_validation"
OUT.mkdir(parents=True, exist_ok=True)

def inspect(path, name, role):
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    nan_num = int(np.isnan(num.to_numpy(dtype=float)).sum()) if not num.empty else 0
    inf_num = int(np.isinf(num.to_numpy(dtype=float)).sum()) if not num.empty else 0
    missing = int(df.isna().sum().sum())
    dup = int(df.duplicated().sum())
    status = "PASS" if missing == 0 and nan_num == 0 and inf_num == 0 else "REVIEW"
    return {
        "dataset": name, "role": role, "file": str(path.relative_to(ROOT)),
        "rows": len(df), "columns": len(df.columns), "missing": missing,
        "numeric_nan": nan_num, "infinite": inf_num, "duplicates": dup,
        "status": status
    }, df

records = []

# 600 MW
paths = list((P / "features" / "600mw").rglob("feature_set_D_full_candidate_pool.csv"))
if paths:
    r, _ = inspect(paths[0], "600 MW", "Power-output forecasting / load-estimation")
    r["notes"] = "Completed 600 MW candidate pool."
    records.append(r)

# HAI
path = P / "features" / "hai" / "hai-23.05" / "model_ready" / "hai_2305_training_model_ready.csv"
if path.exists():
    r, _ = inspect(path, "HAI 23.05", "SCADA anomaly detection / predictive anomaly analysis")
    r["notes"] = "Completed model-ready training dataset."
    records.append(r)

# 6.6 MPa
steam = P / "steam_generator_6_6mpa"
for fn in ["coal_analysis.csv","flue_gas_composition.csv","massflow_temperature.csv",
           "water_steam_circuit.csv","enthalpy_entropy_exergy.csv","operating_condition_summary.csv"]:
    path = steam / fn
    if path.exists():
        r, _ = inspect(path, "6.6 MPa / " + fn[:-4], "Boiler/steam/thermodynamic engineering reference")
        r["notes"] = "Structured engineering table; not a time-series SCADA dataset."
        records.append(r)

summary_path = steam / "operating_condition_summary.csv"
steam_loads = []
if summary_path.exists():
    s = pd.read_csv(summary_path)
    if "mcr_percent" in s:
        steam_loads = s["mcr_percent"].tolist()

# UCI
path = P / "uci_ccpp" / "uci_ccpp_clean.csv"
if path.exists():
    r, df = inspect(path, "UCI CCPP", "Power-generation modeling / supporting validation")
    r["notes"] = "41 duplicate rows retained for review; AT,V,AP,RH predictors and PE target."
    r["status"] = "REVIEW" if r["duplicates"] else r["status"]
    records.append(r)

# Coal metadata
path = P / "coal_plant_metadata.csv"
if path.exists():
    r, df = inspect(path, "Coal Plant Metadata", "Plant-level context / metadata / emissions reference")
    iddup = int(df["id"].duplicated().sum()) if "id" in df else -1
    r["notes"] = f"Plant ID duplicates: {iddup}."
    r["status"] = "REVIEW" if iddup else r["status"]
    records.append(r)

df = pd.DataFrame(records)
df.to_csv(OUT / "dataset_validation_summary.csv", index=False)

registry = pd.DataFrame([
    ["600 MW","Primary ML","Power-output forecasting / load-estimation","Completed"],
    ["HAI 23.05","Primary ML","SCADA anomaly detection / predictive anomaly analysis","Completed"],
    ["6.6 MPa Steam Generator","Supporting engineering","Boiler/steam/thermodynamic reference","Cleaned; validation recorded"],
    ["UCI CCPP","Supporting modeling","Power-generation modeling / supporting validation","Cleaned; duplicate review needed"],
    ["Coal Plant Metadata","Supporting context","Plant metadata / emissions reference","Cleaned"],
], columns=["dataset","category","role","state"])
registry.to_csv(OUT / "dataset_registry.csv", index=False)

report = [
    "AI-SCADA PROJECT DATASET VALIDATION REPORT",
    "="*70,
    "Validation is non-destructive.",
    "",
    df.to_string(index=False),
    "",
    "6.6 MPa MCR conditions found:",
    str(steam_loads),
    "",
    "Expected: [70.73, 92.65, 100.0]",
    "",
    "Project interpretation:",
    "600 MW and HAI 23.05 are the primary ML branches.",
    "6.6 MPa, UCI CCPP and Coal Plant Metadata are supporting datasets.",
    "The datasets should not be blindly merged because their structures and roles differ.",
]
(OUT / "dataset_validation_report.txt").write_text("\n".join(report), encoding="utf-8")

print("="*70)
print("ALL DATASETS VALIDATION")
print("="*70)
print(df[["dataset","rows","columns","missing","infinite","duplicates","status"]].to_string(index=False))
print()
print("6.6 MPa MCR conditions:", steam_loads)
print("Expected: [70.73, 92.65, 100.0]")
print("Validation is non-destructive.")
print("Output:", OUT)
print("="*70)
