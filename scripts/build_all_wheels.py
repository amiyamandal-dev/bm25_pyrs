#!/usr/bin/env python3
"""
Build wheels for all supported platforms and architectures.
This script helps test the build process locally before releasing.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Success: {result.stdout}")
    return True


def build_local_wheel():
    """Build wheel for current platform."""
    print("Building wheel for current platform...")
    return run_command(["maturin", "build", "--release"])


def build_cross_platform_wheels():
    """Build wheels for multiple platforms using cross-compilation."""
    targets = [
        # Linux targets
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "armv7-unknown-linux-gnueabihf",
        # macOS targets (if on macOS)
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
        # Windows targets (if on Windows or with cross-compilation setup)
        # "x86_64-pc-windows-msvc",
        # "i686-pc-windows-msvc",
    ]
    
    success = True
    for target in targets:
        print(f"\nBuilding for target: {target}")
        if not run_command(["maturin", "build", "--release", "--target", target]):
            print(f"Failed to build for {target}")
            success = False
    
    return success


def check_dependencies():
    """Check if required tools are installed."""
    tools = ["maturin", "cargo", "rustc"]
    missing = []
    
    for tool in tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
            print(f"✓ {tool} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(tool)
            print(f"✗ {tool} is not installed")
    
    if missing:
        print(f"\nMissing tools: {', '.join(missing)}")
        print("Please install them before continuing.")
        return False
    
    return True


def main():
    """Main function."""
    print("BM25-RS Wheel Builder")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Build local wheel
    if not build_local_wheel():
        print("Failed to build local wheel")
        sys.exit(1)
    
    # Optionally build cross-platform wheels
    if len(sys.argv) > 1 and sys.argv[1] == "--cross":
        print("\nBuilding cross-platform wheels...")
        if not build_cross_platform_wheels():
            print("Some cross-platform builds failed")
            sys.exit(1)
    
    print("\n✓ All builds completed successfully!")
    print("Wheels are available in: target/wheels/")


if __name__ == "__main__":
    main()