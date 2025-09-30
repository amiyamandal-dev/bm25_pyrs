#!/usr/bin/env python3
"""
Version bumping script for BM25-RS.
Automatically updates version in both Cargo.toml and pyproject.toml.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def get_current_version():
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    content = pyproject_path.read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError("Version not found in pyproject.toml")
    
    return match.group(1)


def bump_version(current_version, bump_type):
    """Bump version based on type."""
    parts = [int(x) for x in current_version.split('.')]
    
    if bump_type == 'major':
        return f"{parts[0] + 1}.0.0"
    elif bump_type == 'minor':
        return f"{parts[0]}.{parts[1] + 1}.0"
    elif bump_type == 'patch':
        return f"{parts[0]}.{parts[1]}.{parts[2] + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def update_version_in_file(file_path, new_version):
    """Update version in a TOML file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Warning: {file_path} not found, skipping")
        return
    
    content = path.read_text()
    # Update the first occurrence of version = "..."
    updated_content = re.sub(
        r'version\s*=\s*"[^"]+"', 
        f'version = "{new_version}"', 
        content, 
        count=1
    )
    
    if content != updated_content:
        path.write_text(updated_content)
        print(f"Updated {file_path} to version {new_version}")
    else:
        print(f"No version found to update in {file_path}")


def run_command(cmd, check=True):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def check_git_status():
    """Check if git working directory is clean."""
    result = run_command("git status --porcelain", check=False)
    if result.returncode == 0 and result.stdout.strip():
        print("Warning: Git working directory is not clean")
        print("Uncommitted changes:")
        print(result.stdout)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Bump version for BM25-RS")
    parser.add_argument(
        'bump_type', 
        choices=['major', 'minor', 'patch'], 
        nargs='?',
        help='Type of version bump'
    )
    parser.add_argument(
        '--version', 
        help='Set specific version (e.g., 1.2.3)'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--commit', 
        action='store_true',
        help='Commit the version changes'
    )
    parser.add_argument(
        '--tag', 
        action='store_true',
        help='Create git tag for the new version'
    )
    parser.add_argument(
        '--push', 
        action='store_true',
        help='Push commits and tags to remote'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.version and not args.bump_type:
        parser.error("Either specify bump_type or --version")
    
    if args.version and args.bump_type:
        parser.error("Cannot specify both bump_type and --version")
    
    try:
        # Get current version
        current_version = get_current_version()
        print(f"Current version: {current_version}")
        
        # Calculate new version
        if args.version:
            new_version = args.version
            # Validate version format
            if not re.match(r'^\d+\.\d+\.\d+$', new_version):
                print("Error: Version must be in format X.Y.Z (e.g., 1.2.3)")
                sys.exit(1)
        else:
            new_version = bump_version(current_version, args.bump_type)
        
        print(f"New version: {new_version}")
        
        if args.dry_run:
            print("\nDry run - no changes made")
            print(f"Would update:")
            print(f"  - pyproject.toml: {current_version} -> {new_version}")
            print(f"  - Cargo.toml: {current_version} -> {new_version}")
            if args.commit:
                print(f"  - Would commit changes")
            if args.tag:
                print(f"  - Would create tag: v{new_version}")
            if args.push:
                print(f"  - Would push to remote")
            return
        
        # Check git status if we're going to commit
        if args.commit and not check_git_status():
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Aborted")
                sys.exit(1)
        
        # Update version files
        update_version_in_file("pyproject.toml", new_version)
        update_version_in_file("Cargo.toml", new_version)
        
        print(f"\n✓ Version updated to {new_version}")
        
        # Git operations
        if args.commit:
            print("Committing changes...")
            run_command("git add pyproject.toml Cargo.toml")
            run_command(f'git commit -m "Bump version to {new_version}"')
            print("✓ Changes committed")
        
        if args.tag:
            tag_name = f"v{new_version}"
            print(f"Creating tag: {tag_name}")
            run_command(f"git tag {tag_name}")
            print(f"✓ Tag {tag_name} created")
        
        if args.push:
            print("Pushing to remote...")
            run_command("git push")
            if args.tag:
                run_command(f"git push origin v{new_version}")
            print("✓ Pushed to remote")
        
        print(f"\n🎉 Version bump completed!")
        
        if not args.commit:
            print("\nNext steps:")
            print(f"  git add pyproject.toml Cargo.toml")
            print(f"  git commit -m 'Bump version to {new_version}'")
            if not args.tag:
                print(f"  git tag v{new_version}")
            if not args.push:
                print(f"  git push && git push --tags")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()