import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data/processed/600mw/600mw_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/features/600mw"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "600mw_feature_categories.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 80)
print("600 MW FEATURE CATEGORIZATION")
print("=" * 80)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# --------------------------------------------------
# 2. Define engineering categories
# --------------------------------------------------

categories = {

    "Power / Electrical": [
        "Power output",
        "Plant auxiliary power",
        "Net power output"
    ],

    "Feedwater": [
        "Feed water flow rate",
        "Feedwater pressure",
        "Feedwater temperature"
    ],

    "Main Steam / Boiler": [
        "Fresh steam pressure",
        "Drum pressure",
        "A-side superheated steam pressure",
        "B-side superheated steam pressure",
        "A-side SH steam temperature",
        "B-side SH steam temperature"
    ],

    "Reheater": [
        "A-side reheater outlet steam temperature",
        "B-side reheater outlet steam temperature",
        "Reheat steam flow rate",
        "Reheater spray water flow rate",
        "Reheater desuperheating water temperature",
        "Reheat steam desuperheating water pressure",
        "A-side reheater emergency spray inlet steam temperature",
        "A-side reheater emergency spray outlet steam temperature",
        "B-side reheater emergency spray inlet steam temperature",
        "B-side reheater emergency spray outlet steam temperature"
    ],

    "Desuperheating / Spray System": [
        "A-side primary desuperheater water flow rate",
        "B-side primary desuperheater water flow rate",
        "A-side secondary desuperheater water flow rate",
        "B-side secondary desuperheater water flow rate",
        "Superheater desuperheating water flow rate",
        "Superheated steam desuperheating water pressure"
    ],

    "Blowdown": [
        "A-side continuous blowdown flow rate",
        "B-side continuous blowdown flow rate"
    ],

    "Air / Combustion": [
        "#1 FD fan inlet air temperature",
        "#2 FD fan inlet air temperature",
        "A-side air preheater outlet secondary air temperature",
        "B-side air preheater outlet secondary air temperature",
        "A-side air heater outlet air temperature",
        "B-side air heater outlet air temperature",
        "A-side air preheater outlet primary air temperature",
        "B-side air preheater outlet primary air temperature",
        "A-side air preheater inlet primary air temperature",
        "B-side air preheater inlet primary air temperature",
        "A-side air preheater outlet primary air temperature 1",
        "A-side air preheater outlet primary air temperature 2",
        "A-side air preheater outlet primary air temperature 3",
        "B-side air preheater outlet primary air temperature 1",
        "B-side air preheater outlet primary air temperature 2",
        "B-side air preheater outlet primary air temperature 3",
        "Air preheater outlet secondary air temperature 1",
        "Air preheater outlet secondary air temperature 2",
        "Air preheater outlet secondary air temperature 3"
    ],

    "Flue Gas": [
        "A-side air preheater outlet flue gas temperature 1",
        "A-side air preheater outlet flue gas temperature 2",
        "A-side air preheater outlet flue gas temperature 3",
        "B-side air preheater outlet flue gas temperature 1",
        "B-side air preheater outlet flue gas temperature 2",
        "B-side air preheater outlet flue gas temperature 3",
        "A-side air preheater outlet flue gas temperature",
        "B-side air preheater outlet flue gas temperature",
        "A-side secondary air preheater outlet flue gas temperature",
        "B-side secondary air preheater outlet flue gas temperature",
        "A-side flue gas oxygen content",
        "B-side flue gas oxygen content"
    ],

    "Coal / Fuel": [
        "#1 coal mill A-side output",
        "#1 coal mill B-side output",
        "#2 coal mill A-side output",
        "#2 coal mill B-side output",
        "#3 coal mill A-side output",
        "#3 coal mill B-side output",
        "#4 coal mill A-side output",
        "#4 coal mill B-side output",
        "#5 coal mill A-side output",
        "#5 coal mill B-side output",
        "#6 coal mill A-side output",
        "#6 coal mill B-side output"
    ],

    "Efficiency / Performance": [
        "Boiler efficiency",
        "Standard coal consumption rate for power generation",
        "turbine heat rate",
        "unit efficiency"
    ]
}


# --------------------------------------------------
# 3. Build feature map
# --------------------------------------------------

rows = []

for category, features in categories.items():

    for feature in features:

        rows.append({
            "feature_name": feature,
            "category": category
        })


feature_map = pd.DataFrame(rows)

# --------------------------------------------------
# Feature-name normalization for validation
# --------------------------------------------------

def normalize_column_name(name):
    name = (
        str(name)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("（", "(")
        .replace("）", ")")
        .strip()
        .lower()
    )

    # Remove measurement units from dataset column names
    units = [
        "(℃)",
        "(mpa)",
        "(mw)",
        "(t/h)",
        "(%)",
        "(g/kwh)",
        "(kj/kwh)"
    ]

    for unit in units:
        name = name.replace(unit, "")

    # Remove extra whitespace
    name = " ".join(name.split())

    return name.strip()
# --------------------------------------------------
# 4. Check that all numeric variables are mapped
# --------------------------------------------------

numeric_columns = [
    column
    for column in df.columns
    if column != "Time"
]

dataset_features = set(numeric_columns)

mapped_features = set(
    feature_map["feature_name"]
)

# Normalize names only for validation.
# Original names are preserved in the saved feature map.

normalized_dataset = {
    normalize_column_name(name): name
    for name in dataset_features
}

normalized_mapped = {
    normalize_column_name(name): name
    for name in mapped_features
}

missing_from_map_normalized = (
    set(normalized_dataset.keys())
    - set(normalized_mapped.keys())
)

extra_in_map_normalized = (
    set(normalized_mapped.keys())
    - set(normalized_dataset.keys())
)

missing_from_map = [
    normalized_dataset[name]
    for name in missing_from_map_normalized
]

extra_in_map = [
    normalized_mapped[name]
    for name in extra_in_map_normalized
]


print("\n" + "-" * 80)
print("CATEGORY VALIDATION")
print("-" * 80)

print(
    "Numeric features in dataset:",
    len(dataset_features)
)

print(
    "Features in category map:",
    len(mapped_features)
)


if missing_from_map:
    print("\nFeatures NOT categorized:")
    for feature in sorted(missing_from_map):
        print("-", feature)
else:
    print("\nAll numeric features categorized.")


if extra_in_map:
    print("\nFeatures in map but NOT dataset:")
    for feature in sorted(extra_in_map):
        print("-", feature)
else:
    print("\nAll mapped features exist in dataset.")


# --------------------------------------------------
# 5. Category summary
# --------------------------------------------------

print("\n" + "-" * 80)
print("CATEGORY SUMMARY")
print("-" * 80)

print(
    feature_map["category"]
    .value_counts()
)


# --------------------------------------------------
# 6. Save feature map
# --------------------------------------------------

feature_map.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("FEATURE CATEGORIZATION COMPLETE")
print("=" * 80)

print("Output:", OUTPUT_FILE)