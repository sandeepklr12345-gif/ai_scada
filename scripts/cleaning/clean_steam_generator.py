import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data/raw/steam_generator_6_6mpa"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/steam_generator_6_6mpa"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("6.6 MPa STEAM GENERATOR DATA CLEANING")
print("=" * 80)


# --------------------------------------------------
# 1. Find Excel file
# --------------------------------------------------

excel_files = sorted(RAW_DIR.glob("*.xlsx"))

print(f"\nExcel files found: {len(excel_files)}")

if len(excel_files) == 0:
    raise FileNotFoundError(
        "No Excel file found in the 6.6 MPa dataset folder."
    )

if len(excel_files) > 1:
    print("\nMultiple Excel files found:")
    for file in excel_files:
        print(" -", file.name)

    raise RuntimeError(
        "More than one Excel file found. Keep only the intended dataset "
        "or select the correct file explicitly."
    )


RAW_FILE = excel_files[0]

print("\nInput file:", RAW_FILE.name)


# --------------------------------------------------
# 2. Read workbook
# --------------------------------------------------

excel = pd.ExcelFile(RAW_FILE)

print("\nSheets found:")
for sheet in excel.sheet_names:
    print(" -", sheet)


# --------------------------------------------------
# 3. Process each sheet separately
# --------------------------------------------------

for sheet in excel.sheet_names:

    print("\n" + "-" * 80)
    print(f"Processing sheet: {sheet}")
    print("-" * 80)

    df = pd.read_excel(
        RAW_FILE,
        sheet_name=sheet
    )

    original_shape = df.shape

    print("Original shape:", original_shape)

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    missing_count = df.isnull().sum().sum()

    print("Total missing values:", missing_count)

    # --------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print("Duplicate rows:", duplicate_count)

    # --------------------------------------------------
    # Data types
    # --------------------------------------------------

    print("\nData types:")
    print(df.dtypes)

    # --------------------------------------------------
    # Columns
    # --------------------------------------------------

    print("\nColumns:")
    print(list(df.columns))

    # --------------------------------------------------
    # Row count verification
    # --------------------------------------------------

    assert len(df) == original_shape[0], (
        "Row count changed unexpectedly."
    )

    # --------------------------------------------------
    # Save cleaned sheet
    # --------------------------------------------------

    safe_sheet_name = (
        sheet
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    output_file = OUTPUT_DIR / f"{safe_sheet_name}.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print("\nSaved:", output_file)


print("\n" + "=" * 80)
print("6.6 MPa STEAM GENERATOR CLEANING COMPLETE")
print("=" * 80)