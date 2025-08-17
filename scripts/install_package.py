#!/usr/bin/env python3
"""
Package installation script for the Unified DTA System
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Command: {command}")
        print(f"  Error: {e.stderr}")
        return None


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is not supported")
        print("  Unified DTA requires Python 3.8 or higher")
        return False
    
    print(f"✓ Python {version.major}.{version.minor} is compatible")
    return True


def install_package(mode="development"):
    """Install the package in development or production mode"""
    
    print("=" * 60)
    print("Unified DTA System - Package Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"Installing from: {project_root}")
    print(f"Installation mode: {mode}")
    
    # Upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install package
    if mode == "development":
        # Development installation with editable mode
        result = run_command(
            f"{sys.executable} -m pip install -e .[dev,docs,visualization,notebooks]",
            "Installing package in development mode"
        )
    else:
        # Production installation
        result = run_command(
            f"{sys.executable} -m pip install .",
            "Installing package in production mode"
        )
    
    if result is None:
        print("\n✗ Package installation failed")
        sys.exit(1)
    
    # Verify installation
    try:
        import unified_dta
        print(f"\n✓ Package installed successfully!")
        print(f"  Version: {unified_dta.__version__}")
        print(f"  Location: {unified_dta.__file__}")
    except ImportError as e:
        print(f"\n✗ Package verification failed: {e}")
        sys.exit(1)
    
    # Test CLI commands
    print("\nTesting CLI commands...")
    cli_commands = [
        "unified-dta --help",
        "dta-train --help",
        "dta-predict --help",
        "dta-evaluate --help"
    ]
    
    for cmd in cli_commands:
        result = run_command(cmd, f"Testing '{cmd.split()[0]}'")
        if result is None:
            print(f"  Warning: CLI command '{cmd.split()[0]}' may not be working properly")
    
    print("\n" + "=" * 60)
    print("Installation Summary:")
    print("=" * 60)
    print("✓ Package installed successfully")
    print("✓ CLI commands registered")
    print("\nNext steps:")
    print("1s.mode)ge(argll_packa insta()
   .parse_args= parserargs       
     )
 
)"ntpmeeveloult: defamode (dion nstallatlp="I
        he,pment"eloault="dev   def
     n"],ctio", "produmentlop["devehoices=      c, 
   "--mode"   ent(
    umargparser.add_  
  ackage")fied DTA ptall Uniiption="Inscrarser(des.ArgumentParseser = argppar    
    se
 argpar  import":
  n__ == "__mai __name__


if0) * 6=" print("
   /'")tstesest -m pyton  with 'pythtests Run t("3.  prin")
   examples for usageentationocum Check the d"2.print(s")
    ndmavailable comp' to see a --hel'unified-dta. Run 