#!/usr/bin/env python3
"""
Package building and distribution script for the Unified DTA System
"""

import subprocess
import sys
import shutil
from pathlib import Path
import os


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(f"  Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Command: {command}")
        print(f"  Error: {e.stderr}")
        return False


def clean_build_artifacts():
    """Clean previous build artifacts"""
    project_root = Path(__file__).parent.parent
    
    artifacts = [
        project_root / "build",
        project_root / "dist", 
        project_root / "unified_dta.egg-info"
    ]
    
    for artifact in artifacts:
        if artifact.exists():
            print(f"Removing {artifact}")
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()


def build_package():
    """Build the package for distribution"""
    
    print("=" * 60)
    print("Unified DTA System - Package Builder")
    print("=" * 60)
    
    # Get project root and change directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"Building from: {project_root}")
    
    # Clean previous builds
    clean_build_artifacts()
    
    # Install build dependencies
    if not run_command(
        f"{sys.executable} -m pip install --upgrade build twine",
        "Installing build dependencies"
    ):
        return False
    
    # Build source distribution
    if not run_command(
        f"{sys.executable} -m build --sdist",
        "Building source distribution"
    ):
        return False
    
    # Build wheel distribution
    if not run_command(
        f"{sys.executable} -m build --wheel", 
        "Building wheel distribution"
    ):
        return False
    
    # List built packages
    dist_dir = project_root / "dist"
    if dist_dir.exists():
        print(f"\n✓ Built packages:")
        for package in dist_dir.iterdir():
            print(f"  - {package.name}")
    
    return True


def check_package():
    """Check the built package for issues"""
    print("\nChecking package...")
    
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("✗ No dist directory found. Run build first.")
        return False
    
    # Check with twine
    return run_command(
        "python -m twine check dist/*",
        "Checking package with twine"
    )


def upload_package(repository="testpypi"):
    """Upload package to PyPI or TestPyPI"""
    
    if repository == "testpypi":
        upload_cmd = "python -m twine upload --repository testpypi dist/*"
        description = "Uploading to TestPyPI"
    else:
        upload_cmd = "python -m twine upload dist/*"
        description = "Uploading to PyPI"
    
    print(f"\n{description}...")
    print("Note: You will need to enter your credentials")
    
    try:
        subprocess.run(upload_cmd, shell=True, check=True)
        print(f"✓ Package uploaded to {repository} successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Upload failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build and distribute Unified DTA package")
    parser.add_argument(
        "action",
        choices=["build", "check", "upload", "all"],
        help="Action to perform"
    )
    parser.add_argument(
        "--repository",
        choices=["testpypi", "pypi"],
        default="testpypi",
        help="Repository for upload (default: testpypi)"
    )
    
    args = parser.parse_args()
    
    success = True
    
    if args.action in ["build", "all"]:
        success = build_package()
    
    if success and args.action in ["check", "all"]:
        success = check_package()
    
    if success and args.action in ["upload", "all"]:
        success = upload_package(args.repository)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ All operations completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Some operations failed. Check the output above.")
        print("=" * 60)
        sys.exit(1)