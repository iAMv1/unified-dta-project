#!/usr/bin/env python3
"""
Reorganize repository into a single root folder with standardized subfolders.

- Creates a new folder: unified_dta_project/
- Subfolders: src/, tests/, configs/, docs/, scripts/, examples/, data/, apps/, external/, tools/, ci/
- Moves existing content into the consolidated layout.
- Skips __pycache__ and *.pyc.
- Handles duplicates by placing second and subsequent collisions under a _dups/ suffix and logging them.
- Supports dry-run (default) and apply mode.

Usage:
  python scripts/utilities/reorganize_project.py --dry-run
  python scripts/utilities/reorganize_project.py --apply

This script is conservative and only uses shutil.move for files/dirs that exist.
It generates a plan first, prints it, and can write a CSV report.
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = REPO_ROOT / "unified_dta_project"

# Mapping of known top-level dirs or nested distributions to consolidated subfolders
DIR_MAP = {
    # Preferred canonical locations
    "src": "src",
    "tests": "tests",
    "configs": "configs",
    "config": "configs",
    "docs": "docs",
    "documentation": "docs",
    "scripts": "scripts",
    "examples": "examples",
    "data": "data",
    "apps": "apps",
    "tools": "tools",
    
    # Alternate roots that should be folded into canonical layout
    "unified-dta-system/src": "src",
    "unified-dta-system/tests": "tests",
    "unified-dta-system/scripts": "scripts",
    "unified-dta-system/docs": "docs",
    "unified-dta-system/examples": "examples",
    "unified-dta-system/configs": "configs",
    "unified-dta-system/data": "data",
    "unified-dta-system/external": "external",

    # Legacy/related projects we vendor under external
    "DeepDTAGen": "external/DeepDTAGen",
    "deepdta_platform": "external/deepdta_platform",
    "DoubleSG-DTA": "external/DoubleSG-DTA",
}

# Top-level files to keep under the new root
TOP_FILES_TO_MOVE = {
    "README.md": "README.md",
    "CHANGELOG.md": "CHANGELOG.md",
    "LICENSE": "LICENSE",
    "Makefile": "Makefile",
    "MANIFEST.in": "MANIFEST.in",
    "requirements.txt": "requirements.txt",
    "requirements-dev.txt": "requirements-dev.txt",
    "setup.py": "setup.py",
    "pyproject.toml": "pyproject.toml",
    
    # Common runner scripts and helpers
    "run_api.py": "scripts/run_api.py",
    "train_2phase.py": "scripts/train_2phase.py",
    "train_combined.py": "scripts/train_combined.py",
    "train_drug_generation.py": "scripts/train_drug_generation.py",
    "prepare_data.py": "scripts/prepare_data.py",
    "verify_unified_project.py": "scripts/verify_unified_project.py",
    "checkpoint_cli.py": "scripts/checkpoint_cli.py",
    "config_cli.py": "scripts/config_cli.py",
    "run_tests.py": "scripts/run_tests.py",
}

SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".github",
    ".venv",
    ".kiro",
    "steering",
}

SKIP_FILE_SUFFIXES = {".pyc", ".pyo"}

def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

def safe_move(src: Path, dst: Path, apply: bool, plan: List[Tuple[Path, Path]]) -> Path:
    dst_parent = dst.parent
    dst_parent.mkdir(parents=True, exist_ok=True)
    final_dst = dst
    counter = 1
    while final_dst.exists():
        # Place duplicates under _dups subfolder with numeric suffix
        dups_parent = dst_parent / "_dups"
        dups_parent.mkdir(parents=True, exist_ok=True)
        final_dst = dups_parent / (dst.name if counter == 1 else f"{dst.stem}_{counter}{dst.suffix}")
        counter += 1
    if apply:
        shutil.move(str(src), str(final_dst))
    plan.append((src, final_dst))
    return final_dst

def gather_moves() -> List[Tuple[Path, Path]]:
    moves: List[Tuple[Path, Path]] = []

    # Ensure target root exists (in dry-run we still simulate path creation)
    TARGET_ROOT.mkdir(exist_ok=True)

    # Map directories
    for key, rel_target in DIR_MAP.items():
        src_path = REPO_ROOT / key
        if not src_path.exists():
            continue
        if any(part in SKIP_DIRS for part in src_path.parts):
            continue
        # Skip if already inside target root
        if is_within(src_path, TARGET_ROOT):
            continue
        target_path = TARGET_ROOT / rel_target
        moves.append((src_path, target_path))

    # Move top-level files
    for fname, rel_target in TOP_FILES_TO_MOVE.items():
        src_file = REPO_ROOT / fname
        if src_file.exists() and src_file.is_file():
            target_file = TARGET_ROOT / rel_target
            moves.append((src_file, target_file))

    # Pick up stray top-level items not covered
    for item in REPO_ROOT.iterdir():
        if item.name in SKIP_DIRS:
            continue
        if is_within(item, TARGET_ROOT):
            continue
        if item.is_dir():
            # Skip directories already mapped explicitly
            mapped = False
            for key in DIR_MAP.keys():
                if (REPO_ROOT / key) == item:
                    mapped = True
                    break
            if mapped:
                continue
            # Put any other top-level dir under external/ or tools/
            dest_group = "external" if any(s in item.name.lower() for s in ["deep", "graph", "double", "dta"]) else "tools"
            moves.append((item, TARGET_ROOT / dest_group / item.name))
        elif item.is_file():
            if item.suffix in SKIP_FILE_SUFFIXES:
                continue
            # Move miscellaneous top-level files under project root misc/
            moves.append((item, TARGET_ROOT / "misc" / item.name))

    return moves


def write_report(moves: List[Tuple[Path, Path]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "destination"])
        for src, dst in moves:
            writer.writerow([str(src), str(dst)])


def main():
    parser = argparse.ArgumentParser(description="Reorganize repo into unified_dta_project root")
    parser.add_argument("--apply", action="store_true", help="Apply the planned moves (default is dry-run)")
    parser.add_argument("--report", type=Path, default=TARGET_ROOT / "reorg_plan.csv", help="Write a CSV plan/report here")
    args = parser.parse_args()

    planned_moves = gather_moves()

    # Execute or print
    print("Planned moves ({}):".format(len(planned_moves)))
    for src, dst in planned_moves:
        print(f" - {src} -> {dst}")

    write_report(planned_moves, args.report)
    print(f"\nPlan written to: {args.report}")

    if args.apply:
        print("\nApplying moves...\n")
        applied: List[Tuple[Path, Path]] = []
        for src, dst in planned_moves:
            if not src.exists():
                print(f"[SKIP] Missing: {src}")
                continue
            if src.is_file() and src.suffix in SKIP_FILE_SUFFIXES:
                print(f"[SKIP] Bytecode: {src}")
                continue
            if any(p in SKIP_DIRS for p in src.parts):
                print(f"[SKIP] In skip list: {src}")
                continue
            final_dst = safe_move(src, dst, apply=True, plan=applied)
            print(f"[MOVED] {src} -> {final_dst}")
        print("\nMove complete.")
        print("IMPORTANT: Update imports and configuration paths as needed. Re-run tests to validate.")
    else:
        print("\nDry-run only. Re-run with --apply to execute.")


if __name__ == "__main__":
    main()

