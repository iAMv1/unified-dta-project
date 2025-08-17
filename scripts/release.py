#!/usr/bin/env python3
"""
Release automation script for the Unified DTA System
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse


def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout.strip():
                print(f"  Output: {result.stdout.strip()}")
        else:
            print(f"✗ {description} failed:")
            print(f"  Error: {result.stderr.strip()}")
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Command: {command}")
        print(f"  Error: {e.stderr}")
        return False


def check_git_status():
    """Check if git working directory is clean"""
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("✗ Git working directory is not clean:")
        print(result.stdout)
        return False
    print("✓ Git working directory is clean")
    return True


def get_current_version():
    """Get current version from package"""
    try:
        import unified_dta
        return unified_dta.__version__
    except ImportError:
        # Fallback to reading from file
        init_file = Path("unified_dta/__init__.py")
        content = init_file.read_text()
        import re
        match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
        raise ValueError("Version not found")


def create_release(version_bump=None, version=None, dry_run=False):
    """Create a new release"""
    
    print("=" * 60)
    print("Unified DTA System - Release Automation")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Check git status
    if not dry_run and not check_git_status():
        print("Please commit or stash your changes before creating a release")
        return False
    
    # Get current version
    try:
        current_version = get_current_version()
        print(f"Current version: {current_version}")
    except Exception as e:
        print(f"✗ Could not get current version: {e}")
        return False
    
    # Update version if requested
    if version_bump or version:
        if version:
            new_version = version
            update_cmd = f"python scripts/update_version.py --version {version}"
        else:
            update_cmd = f"python scripts/update_version.py --increment {version_bump}"
            # We'll get the new version after running the command
            new_version = None
        
        if dry_run:
            print(f"[DRY RUN] Would run: {update_cmd}")
        else:
            if not run_command(update_cmd, "Updating version"):
                return False
            
            # Get the new version
            try:
                new_version = get_current_version()
                print(f"Updated to version: {new_version}")
            except Exception as e:
                print(f"✗ Could not get updated version: {e}")
                return False
    else:
        new_version = current_version
    
    # Run tests
    test_cmd = "python -m pytest tests/ -v"
    if dry_run:
        print(f"[DRY RUN] Would run tests: {test_cmd}")
    else:
        if not run_command(test_cmd, "Running tests"):
            print("Tests failed. Aborting release.")
            return False
    
    # Build package
    build_cmd = "python scripts/build_package.py build"
    if dry_run:
        print(f"[DRY RUN] Would build package: {build_cmd}")
    else:
        if not run_command(build_cmd, "Building package"):
            return False
    
    # Check package
    check_cmd = "python scripts/build_package.py check"
    if dry_run:
        print(f"[DRY RUN] Would check package: {check_cmd}")
    else:
        if not run_command(check_cmd, "Checking package"):
            return False
    
    # Commit version changes (if any)
    if (version_bump or version) and not dry_run:
        commit_cmd = f'git add . && git commit -m "Bump version to {new_version}"'
        if not run_command(commit_cmd, f"Committing version {new_version}"):
            return False
    
    # Create git tag
    tag_name = f"v{new_version}"
    tag_cmd = f'git tag -a {tag_name} -m "Release {tag_name}"'
    if dry_run:
        print(f"[DRY RUN] Would create tag: {tag_cmd}")
    else:
        if not run_command(tag_cmd, f"Creating tag {tag_name}"):
            return False
    
    # Push changes and tags
    if not dry_run:
        if not run_command("git push", "Pushing changes"):
            return False
        if not run_command("git push --tags", "Pushing tags"):
            return False
    else:
        print("[DRY RUN] Would push changes and tags")
    
    print("\n" + "=" * 60)
    print(f"✓ Release {new_version} created successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Upload to PyPI: python scripts/build_package.py upload --repository pypi")
    print("2. Create GitHub release from the tag")
    print("3. Update documentation if needed")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new release")
    
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument(
        "--bump", 
        choices=["major", "minor", "patch"],
        help="Bump version part"
    )
    version_group.add_argument(
        "--version",
        help="Set specific version (e.g., 1.2.3)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    args = parser.parse_args()
    
    try:
        success = create_release(
            version_bump=args.bump,
            version=args.version,
            dry_run=args.dry_run
        )
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nRelease cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)