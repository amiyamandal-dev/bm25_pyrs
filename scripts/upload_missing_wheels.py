#!/usr/bin/env python3
"""
Upload additional wheels for missing platforms to fix PyPI distribution.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return None


def build_source_dist():
    """Build and upload source distribution for better compatibility."""
    print("Building source distribution...")
    
    # Clean and build
    run_command(["maturin", "clean"])
    result = run_command(["maturin", "sdist", "--out", "dist"])
    
    if result and result.returncode == 0:
        # Upload source distribution
        sdist_files = list(Path("dist").glob("*.tar.gz"))
        if sdist_files:
            print(f"Uploading source distribution: {sdist_files[0]}")
            run_command(["maturin", "upload", str(sdist_files[0])])
            return True
    
    return False


def main():
    """Main upload script."""
    print("BM25-RS Missing Wheels Uploader")
    print("=" * 40)
    
    print("🔧 Strategy: Upload source distribution for pip to build locally")
    print("   This allows pip to build wheels for any platform from source")
    print()
    
    if build_source_dist():
        print("✅ Source distribution uploaded successfully!")
        print()
        print("📋 Users can now install on any platform with:")
        print("   pip install bm25-rs==0.1.1")
        print()
        print("   Pip will automatically build from source if no wheel is available")
        print("   for their platform.")
    else:
        print("❌ Failed to upload source distribution")
        sys.exit(1)


if __name__ == "__main__":
    main()