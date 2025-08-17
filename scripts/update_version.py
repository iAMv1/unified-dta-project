#!/usr/bin/env python3
"""
Version management script for the Unified DTA System
"""

import re
import sys
from pathlib import Path
from typing import Tuple


def get_current_version() -> str:
    """Get current version from __init__.py"""
    init_file = Path("unified_dta/__init__.py")
    content = init_file.read_text()
    match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    raise ValueError("Version not found in __init__.py")


def parse_version(version: str) -> Tuple[int, int, int]:
    """Parse version string into components"""
    parts = version.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version}")
    return tuple(int(part) for part in parts)


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version components into string"""
    return f"{major}.{minor}.{patch}"


def update_version(new_version: str):
    """Update version in __init__.py"""
    init_file = Path("unified_dta/__init__.py")
    content = init_file.read_text()
    
    # Update version
    new_content = re.sub(
        r'__version__ = ["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        content
    )
    
    init_file.write_text(new_content)
    print(f"Updated version to {new_version}")


def bump_version(bump_type: str):
    """Bump version by type (major, minor, patch)"""
    current = get_current_version()
    major, minor, patch = parse_version(current)
    
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    
    new_version = format_version(major, minor, patch)
    update_version(new_version)
    return new_version


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python update_version.py <bump_type|version>")
        print("  bump_type: major, minor, patch")
        print("  version: specific version like 1.2.3")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    try:
        current = get_current_version()
        print(f"Current version: {current}")
        
        if arg in ["major", "minor", "patch"]:
            new_version = bump_version(arg)
        else:
            # Assume it's a specific version
            parse_version(arg)  # Validate format
            update_version(arg)
            new_version = arg
        
        print(f"New version: {new_version}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()