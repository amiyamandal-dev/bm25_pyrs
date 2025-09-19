#!/usr/bin/env python3
"""
Cross-platform wheel building script for BM25-RS.
"""

import os
import sys
import subprocess
import platform
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


def clean_build_artifacts():
    """Clean previous build artifacts."""
    print("Cleaning build artifacts...")
    
    dirs_to_clean = ["dist", "target", "build", "*.egg-info"]
    
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                print(f"Removing directory: {path}")
                import shutil
                shutil.rmtree(path)


def check_maturin():
    """Check if maturin is installed and up to date."""
    try:
        result = run_command(["maturin", "--version"], check=False)
        if result.returncode != 0:
            print("Installing maturin...")
            run_command([sys.executable, "-m", "pip", "install", "maturin[patchelf]"])
        else:
            print(f"Maturin available: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Installing maturin...")
        run_command([sys.executable, "-m", "pip", "install", "maturin[patchelf]"])


def build_wheels_local():
    """Build wheels for the current platform."""
    print(f"Building wheels for {platform.system()} {platform.machine()}...")
    
    # Build with interpreter discovery
    run_command([
        "maturin", "build", 
        "--release", 
        "--find-interpreter",
        "--out", "dist"
    ])


def build_wheels_cross_platform():
    """Build wheels for multiple platforms using Docker (Linux only)."""
    if platform.system() != "Linux":
        print("Cross-platform building only supported on Linux")
        return
    
    print("Building cross-platform wheels...")
    
    # Check if Docker is available
    try:
        run_command(["docker", "--version"], check=False)
    except FileNotFoundError:
        print("Docker not available, skipping cross-platform builds")
        return
    
    # Build for multiple architectures
    architectures = ["x86_64", "aarch64"]
    
    for arch in architectures:
        print(f"Building for {arch}...")
        try:
            run_command([
                "maturin", "build", 
                "--release",
                "--target", arch,
                "--manylinux", "auto",
                "--out", "dist"
            ])
        except Exception as e:
            print(f"Failed to build for {arch}: {e}")
            continue


def build_source_distribution():
    """Build source distribution."""
    print("Building source distribution...")
    run_command(["maturin", "sdist", "--out", "dist"])


def verify_distributions():
    """Verify that distributions were built correctly."""
    print("Verifying distributions...")
    
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("Error: dist directory not found")
        return False
    
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    
    print(f"Found {len(wheels)} wheel(s):")
    for wheel in wheels:
        print(f"  - {wheel.name}")
    
    print(f"Found {len(sdists)} source distribution(s):")
    for sdist in sdists:
        print(f"  - {sdist.name}")
    
    if not wheels and not sdists:
        print("Error: No distributions found")
        return False
    
    return True


def test_wheel_installation():
    """Test that wheels can be installed."""
    print("Testing wheel installation...")
    
    dist_dir = Path("dist")
    wheels = list(dist_dir.glob("*.whl"))
    
    if not wheels:
        print("No wheels to test")
        return
    
    # Test the first wheel found
    wheel = wheels[0]
    print(f"Testing installation of {wheel.name}")
    
    try:
        # Create a temporary virtual environment
        import tempfile
        import venv
        
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir = Path(temp_dir) / "test_env"
            venv.create(venv_dir, with_pip=True)
            
            # Get the python executable in the venv
            if platform.system() == "Windows":
                python_exe = venv_dir / "Scripts" / "python.exe"
            else:
                python_exe = venv_dir / "bin" / "python"
            
            # Install the wheel
            run_command([str(python_exe), "-m", "pip", "install", str(wheel)])
            
            # Test import
            run_command([
                str(python_exe), "-c", 
                "from bm25_rs import BM25Okapi; print('✅ Import successful')"
            ])
            
            print("✅ Wheel installation test passed")
    
    except Exception as e:
        print(f"❌ Wheel installation test failed: {e}")


def main():
    """Main build script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build BM25-RS wheels")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--local-only", action="store_true", help="Build only for current platform")
    parser.add_argument("--cross-platform", action="store_true", help="Build for multiple platforms")
    parser.add_argument("--test", action="store_true", help="Test wheel installation")
    parser.add_argument("--all", action="store_true", help="Build everything")
    
    args = parser.parse_args()
    
    if not any([args.clean, args.local_only, args.cross_platform, args.test, args.all]):
        args.local_only = True  # Default action
    
    print("BM25-RS Cross-Platform Wheel Builder")
    print("=" * 40)
    
    if args.clean or args.all:
        clean_build_artifacts()
    
    if args.local_only or args.all:
        check_maturin()
        build_wheels_local()
        build_source_distribution()
    
    if args.cross_platform or args.all:
        check_maturin()
        build_wheels_cross_platform()
    
    if verify_distributions():
        print("✅ Build completed successfully!")
        
        if args.test or args.all:
            test_wheel_installation()
    else:
        print("❌ Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()