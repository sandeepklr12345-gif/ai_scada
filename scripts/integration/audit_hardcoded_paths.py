from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCAN_EXTENSIONS = {
    ".py",
    ".json",
    ".txt",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
}

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}

WINDOWS_ABSOLUTE_PATTERN = re.compile(
    r"[A-Za-z]:[\\/][^\"'\n\r]+"
)

def should_scan(path: Path) -> bool:
    if path.suffix.lower() not in SCAN_EXTENSIONS:
        return False

    return not any(part in EXCLUDED_DIRS for part in path.parts)


def scan_file(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return []

    matches = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if WINDOWS_ABSOLUTE_PATTERN.search(line):
            matches.append((line_number, line.strip()))

    return matches


def main():
    print("=" * 80)
    print("HARDCODED PATH AUDIT")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print()

    total_files = 0
    files_with_paths = 0
    total_matches = 0

    for path in sorted(PROJECT_ROOT.rglob("*")):

        if not path.is_file():
            continue

        if not should_scan(path):
            continue

        total_files += 1

        matches = scan_file(path)

        if not matches:
            continue

        files_with_paths += 1
        total_matches += len(matches)

        relative_path = path.relative_to(PROJECT_ROOT)

        print(f"\nFILE: {relative_path}")

        for line_number, line in matches:
            print(f"  Line {line_number}: {line}")

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    print(f"Files scanned          : {total_files}")
    print(f"Files containing paths : {files_with_paths}")
    print(f"Total path occurrences : {total_matches}")
    print("=" * 80)


if __name__ == "__main__":
    main()