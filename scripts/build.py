#!/usr/bin/env python3
"""
Build script for BM25-RS package.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        sys.exit(1)


def clean_build():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    
    dirs_to_clean = [
        "target",
        "dist", 
        "build",
        "*.egg-info",
        "__pycache__",
        ".pytest_cache"
    ]
    
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                print(f"Removing directory: {path}")
                shutil.rmtree(path)
            elif path.is_file():
                print(f"Removing file: {path}")
                path.unlink()


def check_dependencies():
    """Check that required dependencies are available."""
    print("Checking dependencies...")
    
    # Check for Rust
    try:
        result = run_command(["rustc", "--version"], check=False)
        if result.returncode != 0:
            print("Error: Rust compiler not found. Install from https://rustup.rs/")
            sys.exit(1)
        print(f"Rust: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Error: Rust compiler not found. Install from https://rustup.rs/")
        sys.exit(1)
    
    # Check for maturin
    try:
        result = run_command(["maturin", "--version"], check=False)
        if result.returncode != 0:
            print("Error: maturin not found. Install with: pip install maturin")
            sys.exit(1)
        print(f"Maturin: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Error: maturin not found. Install with: pip install maturin")
        sys.exit(1)


def build_development():
    """Build for development."""
    print("Building for development...")
    env = os.environ.copy()
    env["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] = "1"
    result = subprocess.run(["maturin", "develop", "--release"], env=env, check=True)
    return result


def build_wheel():
    """Build wheel for distribution."""
    print("Building wheel...")
    env = os.environ.copy()
    env["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] = "1"
    result = subprocess.run(["maturin", "build", "--release"], env=env, check=True)
    return result


def build_sdist():
    """Build source distribution."""
    print("Building source distribution...")
    run_command(["maturin", "sdist"])


def run_tests():
    """Run the test suite."""
    print("Running tests...")
    
    # Check if pytest is available
    try:
        run_command(["python", "-m", "pytest", "--version"], check=False)
    except:
        print("Installing pytest...")
        run_command(["pip", "install", "pytest"])
    
    # Run tests
    run_command(["python", "-m", "pytest", "tests/", "-v"])


def run_benchmarks():
    """Run benchmarks."""
    print("Running benchmarks...")
    
    # Run basic benchmark
    run_command(["python", "examples/performance_demo.py"])


def main():
    """Main build script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build BM25-RS package")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--dev", action="store_true", help="Build for development")
    parser.add_argument("--wheel", action="store_true", help="Build wheel")
    parser.add_argument("--sdist", action="store_true", help="Build source distribution")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--all", action="store_true", help="Build everything")
    
    args = parser.parse_args()
    
    if not any([args.clean, args.dev, args.wheel, args.sdist, args.test, args.benchmark, args.all]):
        # Default action
        args.dev = True
    
    print("BM25-RS Build Script")
    print("=" * 30)
    
    if args.clean or args.all:
        clean_build()
    
    if args.dev or args.wheel or args.sdist or args.all:
        check_dependencies()
    
    if args.dev or args.all:
        build_development()
    
    if args.wheel or args.all:
        build_wheel()
    
    if args.sdist or args.all:
        build_sdist()
    
    if args.test or args.all:
        run_tests()
    
    if args.benchmark or args.all:
        run_benchmarks()
    
    print("\nBuild completed successfully!")


if __name__ == "__main__":
    main()