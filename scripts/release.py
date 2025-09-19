#!/usr/bin/env python3
"""
Release script for BM25-RS package.
"""

import os
import sys
import subprocess
import re
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
        sys.exit(1)


def get_version():
    """Get version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        sys.exit(1)
    
    content = pyproject_path.read_text()
    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not version_match:
        print("Error: Version not found in pyproject.toml")
        sys.exit(1)
    
    return version_match.group(1)


def check_git_status():
    """Check git status and ensure clean working directory."""
    print("Checking git status...")
    
    # Check if we're in a git repository
    try:
        run_command(["git", "status", "--porcelain"])
    except:
        print("Error: Not in a git repository")
        sys.exit(1)
    
    # Check for uncommitted changes
    result = run_command(["git", "status", "--porcelain"], check=False)
    if result.stdout.strip():
        print("Error: Working directory is not clean. Commit or stash changes first.")
        print("Uncommitted changes:")
        print(result.stdout)
        sys.exit(1)
    
    print("Git working directory is clean.")


def run_tests():
    """Run comprehensive tests before release."""
    print("Running comprehensive tests...")
    
    # Run unit tests
    run_command(["python", "-m", "pytest", "tests/", "-v"])
    
    # Run examples to ensure they work
    examples_dir = Path("examples")
    if examples_dir.exists():
        for example in examples_dir.glob("*.py"):
            print(f"Running example: {example}")
            run_command(["python", str(example)])


def build_distributions():
    """Build wheel and source distributions."""
    print("Building distributions...")
    
    # Clean previous builds
    for dist_dir in ["dist", "target/wheels"]:
        dist_path = Path(dist_dir)
        if dist_path.exists():
            import shutil
            shutil.rmtree(dist_path)
    
    # Build wheel and sdist
    run_command(["maturin", "build", "--release"])
    run_command(["maturin", "sdist"])


def create_git_tag(version):
    """Create git tag for the release."""
    tag_name = f"v{version}"
    
    print(f"Creating git tag: {tag_name}")
    
    # Check if tag already exists
    result = run_command(["git", "tag", "-l", tag_name], check=False)
    if result.stdout.strip():
        print(f"Warning: Tag {tag_name} already exists")
        response = input("Do you want to delete and recreate it? (y/N): ")
        if response.lower() == 'y':
            run_command(["git", "tag", "-d", tag_name])
        else:
            print("Skipping tag creation")
            return
    
    # Create annotated tag
    run_command(["git", "tag", "-a", tag_name, "-m", f"Release {version}"])
    print(f"Created tag: {tag_name}")


def upload_to_pypi(test=False):
    """Upload distributions to PyPI."""
    if test:
        print("Uploading to Test PyPI...")
        repository = "testpypi"
        repository_url = "https://test.pypi.org/legacy/"
    else:
        print("Uploading to PyPI...")
        repository = "pypi"
        repository_url = "https://upload.pypi.org/legacy/"
    
    # Check if twine is available
    try:
        run_command(["twine", "--version"], check=False)
    except FileNotFoundError:
        print("Installing twine...")
        run_command(["pip", "install", "twine"])
    
    # Find distribution files
    dist_files = []
    for pattern in ["dist/*.whl", "dist/*.tar.gz", "target/wheels/*.whl"]:
        dist_files.extend(Path(".").glob(pattern))
    
    if not dist_files:
        print("Error: No distribution files found")
        sys.exit(1)
    
    print(f"Found distribution files: {[str(f) for f in dist_files]}")
    
    # Upload
    cmd = ["twine", "upload"]
    if test:
        cmd.extend(["--repository", repository])
    cmd.extend([str(f) for f in dist_files])
    
    run_command(cmd)


def push_to_git():
    """Push commits and tags to git remote."""
    print("Pushing to git remote...")
    
    # Push commits
    run_command(["git", "push"])
    
    # Push tags
    run_command(["git", "push", "--tags"])


def main():
    """Main release script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Release BM25-RS package")
    parser.add_argument("--test", action="store_true", help="Upload to Test PyPI instead of PyPI")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-git-checks", action="store_true", help="Skip git status checks")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading to PyPI")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without uploading")
    
    args = parser.parse_args()
    
    print("BM25-RS Release Script")
    print("=" * 30)
    
    # Get version
    version = get_version()
    print(f"Releasing version: {version}")
    
    # Confirm release
    if not args.dry_run:
        target = "Test PyPI" if args.test else "PyPI"
        response = input(f"Are you sure you want to release version {version} to {target}? (y/N): ")
        if response.lower() != 'y':
            print("Release cancelled")
            sys.exit(0)
    
    # Check git status
    if not args.skip_git_checks:
        check_git_status()
    
    # Run tests
    if not args.skip_tests:
        run_tests()
    
    # Build distributions
    build_distributions()
    
    if args.dry_run:
        print("Dry run completed. Distributions built but not uploaded.")
        return
    
    # Create git tag
    if not args.skip_git_checks:
        create_git_tag(version)
    
    # Upload to PyPI
    if not args.skip_upload:
        upload_to_pypi(test=args.test)
    
    # Push to git
    if not args.skip_git_checks:
        push_to_git()
    
    print(f"\nRelease {version} completed successfully!")
    
    if args.test:
        print("You can install the test version with:")
        print(f"pip install --index-url https://test.pypi.org/simple/ bm25-rs=={version}")
    else:
        print("You can install the released version with:")
        print(f"pip install bm25-rs=={version}")


if __name__ == "__main__":
    main()