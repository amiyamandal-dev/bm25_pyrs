# Release Guide for BM25-RS

This guide explains how to release new versions of the BM25-RS package to PyPI.

## Prerequisites

1. **GitHub Repository Setup**:
   - Ensure you have write access to the repository
   - Set up PyPI trusted publishing (see below)

2. **Local Development Setup**:
   - Install Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
   - Install Python dependencies: `pip install maturin twine pytest`
   - Install the package in development mode: `maturin develop`

## PyPI Trusted Publishing Setup

To enable secure publishing to PyPI without API tokens:

1. Go to [PyPI Trusted Publishers](https://pypi.org/manage/account/publishing/)
2. Add a new trusted publisher with these settings:
   - **PyPI Project Name**: `bm25-rs`
   - **Owner**: `dorianbrown` (or your GitHub username)
   - **Repository name**: `rank_bm25`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`

For Test PyPI, do the same at [Test PyPI Trusted Publishers](https://test.pypi.org/manage/account/publishing/) with environment name `testpypi`.

## Release Process

### Method 1: Fully Automated Release (Recommended)

1. **Trigger version bump and release via GitHub Actions**:
   - Go to [Actions tab](https://github.com/dorianbrown/rank_bm25/actions)
   - Select "Version Bump" workflow
   - Click "Run workflow"
   - Choose bump type (patch/minor/major) or enter custom version
   - Check "Create release after version bump" for automatic release

2. **Monitor the process**:
   - Version Bump workflow updates versions and creates tag
   - Release workflow automatically triggers and builds all platforms
   - Packages are uploaded to PyPI

### Method 2: Local Version Management + Auto Release

1. **Bump version locally**:
   ```bash
   # Patch version (0.1.2 -> 0.1.3)
   python scripts/bump_version.py patch --commit --tag --push
   
   # Minor version (0.1.2 -> 0.2.0)
   python scripts/bump_version.py minor --commit --tag --push
   
   # Major version (0.1.2 -> 1.0.0)
   python scripts/bump_version.py major --commit --tag --push
   
   # Custom version
   python scripts/bump_version.py --version 0.1.5 --commit --tag --push
   ```

2. **Release automatically triggers**:
   - Pushing the tag triggers the Release workflow
   - Builds and uploads to PyPI automatically

### Method 3: Manual Release (Legacy)

1. **Run the release script**:
   ```bash
   python scripts/release.py
   ```
   
   This script will:
   - Check git status (ensure clean working directory)
   - Run comprehensive tests
   - Build and test wheels locally
   - Create a git tag
   - Push changes and tags to GitHub

2. **Monitor the GitHub Actions**:
   - Go to [Actions tab](https://github.com/dorianbrown/rank_bm25/actions)
   - Watch the "Release" workflow
   - It will build wheels for all platforms and upload to PyPI

### Method 2: Manual Release

1. **Update versions manually**:
   ```bash
   # Edit Cargo.toml
   version = "0.1.3"
   
   # Edit pyproject.toml
   version = "0.1.3"
   ```

2. **Test locally**:
   ```bash
   # Run tests
   python -m pytest tests/ -v
   
   # Build wheel
   maturin build --release
   
   # Test the built wheel
   pip install target/wheels/bm25_rs-*.whl
   python -c "import bm25_rs; print('Import successful')"
   ```

3. **Commit and tag**:
   ```bash
   git add Cargo.toml pyproject.toml
   git commit -m "Bump version to 0.1.3"
   git tag v0.1.3
   git push origin main
   git push origin v0.1.3
   ```

4. **GitHub Actions will automatically**:
   - Build wheels for all supported platforms
   - Upload to PyPI using trusted publishing

## Testing Releases

### Test on Test PyPI

1. **Manual test upload**:
   ```bash
   python scripts/release.py --test
   ```

2. **Using GitHub Actions**:
   - Go to Actions → "Build Test"
   - Click "Run workflow"
   - Select platforms to build
   - Check "Upload to Test PyPI"

3. **Install from Test PyPI**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ bm25-rs==0.1.3
   ```

### Local Cross-Platform Testing

```bash
# Build for current platform
python scripts/build_all_wheels.py

# Build for multiple platforms (requires cross-compilation setup)
python scripts/build_all_wheels.py --cross
```

## Supported Platforms

The release workflow builds wheels for:

### Linux (manylinux)
- x86_64 (Intel/AMD 64-bit)
- x86 (Intel/AMD 32-bit)
- aarch64 (ARM 64-bit)
- armv7 (ARM 32-bit)
- s390x (IBM System z)
- ppc64le (PowerPC 64-bit Little Endian)

### Linux (musllinux)
- x86_64-unknown-linux-musl
- aarch64-unknown-linux-musl

### Windows
- x64 (64-bit)
- x86 (32-bit)

### macOS
- x86_64 (Intel)
- aarch64 (Apple Silicon M1/M2)

## Troubleshooting

### Build Failures

1. **Rust compilation errors**:
   - Check Rust toolchain version
   - Ensure all dependencies are compatible
   - Review Cargo.toml for version conflicts

2. **Python binding errors**:
   - Verify PyO3 version compatibility
   - Check Python version support in pyproject.toml

3. **Cross-compilation issues**:
   - Some targets may require additional setup
   - Check GitHub Actions logs for specific errors

### Upload Failures

1. **PyPI authentication**:
   - Ensure trusted publishing is set up correctly
   - Check repository and environment names match

2. **Version conflicts**:
   - PyPI doesn't allow re-uploading the same version
   - Increment version number and try again

3. **File size limits**:
   - PyPI has file size limits (100MB per file, 60GB per project)
   - Consider optimizing binary size if needed

## Post-Release Checklist

1. **Verify installation**:
   ```bash
   pip install bm25-rs==0.1.3
   python -c "import bm25_rs; print(bm25_rs.__version__)"
   ```

2. **Update documentation**:
   - Update CHANGELOG.md
   - Update version references in README.md
   - Update any version-specific documentation

3. **Create GitHub Release**:
   - The workflow automatically creates a GitHub release
   - Add release notes if needed
   - Attach additional files if necessary

4. **Announce the release**:
   - Update project documentation
   - Notify users through appropriate channels

## Emergency Procedures

### Yanking a Release

If a critical issue is found after release:

1. **Yank from PyPI**:
   ```bash
   twine yank bm25-rs 0.1.3 -m "Critical bug in version 0.1.3"
   ```

2. **Fix the issue and release a patch version**:
   ```bash
   # Fix the bug, then release 0.1.4
   python scripts/release.py
   ```

### Rolling Back

If you need to revert changes:

1. **Revert git commits**:
   ```bash
   git revert <commit-hash>
   git push origin main
   ```

2. **Delete problematic tag**:
   ```bash
   git tag -d v0.1.3
   git push origin :refs/tags/v0.1.3
   ```

## Version Management Tools

### GitHub Actions Workflows

1. **Version Bump Workflow** (`.github/workflows/version-bump.yml`):
   - Automatically updates version in both `Cargo.toml` and `pyproject.toml`
   - Creates git tag
   - Optionally triggers release workflow
   - Supports patch/minor/major bumps or custom versions

2. **Release Workflow** (`.github/workflows/release.yml`):
   - Builds wheels for all supported platforms
   - Uploads to PyPI using trusted publishing
   - Creates GitHub releases
   - Can be triggered by tags or manual dispatch

3. **Build Test Workflow** (`.github/workflows/build-test.yml`):
   - Test builds for specific platforms
   - Optional upload to Test PyPI

### Local Scripts

1. **Version Bump Script** (`scripts/bump_version.py`):
   ```bash
   # Basic usage
   python scripts/bump_version.py patch
   python scripts/bump_version.py minor
   python scripts/bump_version.py major
   python scripts/bump_version.py --version 1.0.0
   
   # With git operations
   python scripts/bump_version.py patch --commit --tag --push
   
   # Dry run to see what would happen
   python scripts/bump_version.py patch --dry-run
   ```

2. **Build Script** (`scripts/build_all_wheels.py`):
   ```bash
   # Build for current platform
   python scripts/build_all_wheels.py
   
   # Build for multiple platforms (requires setup)
   python scripts/build_all_wheels.py --cross
   ```

3. **Release Script** (`scripts/release.py`):
   ```bash
   # Full release process
   python scripts/release.py
   
   # Test release to Test PyPI
   python scripts/release.py --test
   
   # Skip tests (not recommended)
   python scripts/release.py --skip-tests
   ```

## Quick Reference

### Common Release Commands

```bash
# Quick patch release (most common)
python scripts/bump_version.py patch --commit --tag --push

# Quick minor release
python scripts/bump_version.py minor --commit --tag --push

# Test what a version bump would do
python scripts/bump_version.py patch --dry-run

# Manual version with full automation
python scripts/bump_version.py --version 1.0.0 --commit --tag --push

# Build and test locally before release
python scripts/build_all_wheels.py
python -m pytest tests/ -v
```

### GitHub Actions Quick Actions

1. **Automated Release**: Actions → Version Bump → Run workflow → Select bump type → Check "Create release"
2. **Test Build**: Actions → Build Test → Run workflow → Select platforms
3. **Manual Release**: Actions → Release → Run workflow → Enter version

## Additional Resources

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Documentation](https://maturin.rs/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Semantic Versioning](https://semver.org/)