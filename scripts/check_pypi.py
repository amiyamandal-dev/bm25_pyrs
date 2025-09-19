#!/usr/bin/env python3
"""
Check PyPI status and troubleshoot publication issues.
"""

import requests
import json
import sys
from pathlib import Path


def check_package_on_pypi(package_name):
    """Check if package exists on PyPI and get version info."""
    print(f"Checking PyPI for package: {package_name}")
    
    try:
        # Check PyPI API
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Package '{package_name}' found on PyPI")
            
            # Get version info
            versions = list(data["releases"].keys())
            latest_version = data["info"]["version"]
            
            print(f"📦 Latest version: {latest_version}")
            print(f"📋 All versions: {', '.join(sorted(versions, reverse=True)[:10])}")
            
            # Check available files for latest version
            latest_files = data["releases"].get(latest_version, [])
            if latest_files:
                print(f"\n📁 Files for version {latest_version}:")
                for file_info in latest_files:
                    filename = file_info["filename"]
                    python_version = file_info.get("python_version", "unknown")
                    platform = file_info.get("platform_tag", "unknown")
                    print(f"  - {filename} (Python: {python_version}, Platform: {platform})")
            else:
                print(f"❌ No files found for version {latest_version}")
            
            return True, data
            
        elif response.status_code == 404:
            print(f"❌ Package '{package_name}' not found on PyPI")
            return False, None
        else:
            print(f"❌ Error checking PyPI: HTTP {response.status_code}")
            return False, None
            
    except requests.RequestException as e:
        print(f"❌ Network error checking PyPI: {e}")
        return False, None


def check_test_pypi(package_name):
    """Check if package exists on Test PyPI."""
    print(f"\nChecking Test PyPI for package: {package_name}")
    
    try:
        response = requests.get(f"https://test.pypi.org/pypi/{package_name}/json", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Package '{package_name}' found on Test PyPI")
            
            versions = list(data["releases"].keys())
            latest_version = data["info"]["version"]
            
            print(f"📦 Latest version: {latest_version}")
            print(f"📋 All versions: {', '.join(sorted(versions, reverse=True)[:5])}")
            
            return True, data
        else:
            print(f"❌ Package '{package_name}' not found on Test PyPI")
            return False, None
            
    except requests.RequestException as e:
        print(f"❌ Network error checking Test PyPI: {e}")
        return False, None


def get_local_version():
    """Get version from local pyproject.toml."""
    try:
        import tomllib
        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        
        version = config["project"]["version"]
        name = config["project"]["name"]
        
        print(f"📋 Local package: {name} v{version}")
        return name, version
        
    except Exception as e:
        print(f"❌ Error reading local version: {e}")
        return None, None


def check_wheel_compatibility():
    """Check local wheel files for platform compatibility."""
    print("\n🔍 Checking local wheel files...")
    
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ No dist directory found. Run build first.")
        return
    
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        print("❌ No wheel files found in dist/")
        return
    
    print(f"📁 Found {len(wheels)} wheel file(s):")
    
    platforms = set()
    python_versions = set()
    
    for wheel in wheels:
        parts = wheel.stem.split("-")
        if len(parts) >= 5:
            package_name = parts[0]
            version = parts[1]
            python_tag = parts[2]
            abi_tag = parts[3]
            platform_tag = parts[4]
            
            platforms.add(platform_tag)
            python_versions.add(python_tag)
            
            print(f"  - {wheel.name}")
            print(f"    Python: {python_tag}, ABI: {abi_tag}, Platform: {platform_tag}")
    
    print(f"\n📊 Summary:")
    print(f"  Python versions: {', '.join(sorted(python_versions))}")
    print(f"  Platforms: {', '.join(sorted(platforms))}")
    
    # Check for common platforms
    expected_platforms = ["win_amd64", "win32", "macosx_", "linux_x86_64", "linux_aarch64"]
    missing_platforms = []
    
    for expected in expected_platforms:
        if not any(expected in p for p in platforms):
            missing_platforms.append(expected)
    
    if missing_platforms:
        print(f"⚠️  Missing platforms: {', '.join(missing_platforms)}")
        print("   Consider building cross-platform wheels for better compatibility")


def suggest_fixes():
    """Suggest fixes for common issues."""
    print("\n🔧 Troubleshooting Suggestions:")
    print()
    
    print("1. **Version not found on PyPI:**")
    print("   - Check if the release process completed successfully")
    print("   - Verify GitHub Actions workflow ran without errors")
    print("   - Check PyPI upload logs for authentication issues")
    print()
    
    print("2. **Cross-platform compatibility:**")
    print("   - Use GitHub Actions to build wheels for all platforms")
    print("   - Run: python scripts/build_wheels.py --cross-platform")
    print("   - Ensure maturin-action is used in CI/CD")
    print()
    
    print("3. **Authentication issues:**")
    print("   - Verify PyPI API token is correctly configured")
    print("   - Check if trusted publishing is set up correctly")
    print("   - Try manual upload: maturin upload dist/*")
    print()
    
    print("4. **Build issues:**")
    print("   - Clean build: python scripts/build.py --clean")
    print("   - Rebuild: python scripts/build_wheels.py --all")
    print("   - Check Rust toolchain: rustc --version")
    print()
    
    print("5. **Test before release:**")
    print("   - Upload to Test PyPI first: python scripts/release.py --test")
    print("   - Test installation: pip install -i https://test.pypi.org/simple/ bm25-rs")


def main():
    """Main check script."""
    print("BM25-RS PyPI Status Checker")
    print("=" * 40)
    
    # Get local version
    package_name, local_version = get_local_version()
    
    if not package_name:
        print("❌ Could not determine local package info")
        sys.exit(1)
    
    # Check PyPI
    pypi_exists, pypi_data = check_package_on_pypi(package_name)
    
    # Check Test PyPI
    test_pypi_exists, test_pypi_data = check_test_pypi(package_name)
    
    # Check local wheels
    check_wheel_compatibility()
    
    # Analysis
    print("\n📊 Analysis:")
    
    if pypi_exists and pypi_data:
        pypi_version = pypi_data["info"]["version"]
        if local_version == pypi_version:
            print(f"✅ Local version {local_version} matches PyPI")
        elif local_version in pypi_data["releases"]:
            print(f"✅ Local version {local_version} exists on PyPI")
        else:
            print(f"❌ Local version {local_version} not found on PyPI")
            print(f"   Latest PyPI version: {pypi_version}")
    else:
        print(f"❌ Package not found on PyPI")
    
    if not pypi_exists and not test_pypi_exists:
        print("❌ Package not found on either PyPI or Test PyPI")
        print("   This suggests the package has never been published")
    
    # Suggestions
    suggest_fixes()


if __name__ == "__main__":
    main()