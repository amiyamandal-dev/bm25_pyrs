# Release Workflow Guide

This repository has automated workflows for version bumping, building, and releasing to PyPI.

## 🚀 Quick Release (Recommended)

For most releases, use the **Quick Release** workflow:

1. Go to **Actions** → **Quick Release**
2. Click **Run workflow**
3. Select version bump type:
   - `patch`: 1.0.0 → 1.0.1 (bug fixes)
   - `minor`: 1.0.0 → 1.1.0 (new features)
   - `major`: 1.0.0 → 2.0.0 (breaking changes)
4. Click **Run workflow**

This will automatically:
- Bump the version in `pyproject.toml` and `Cargo.toml`
- Create and push a git tag
- Build wheels for all platforms
- Publish to PyPI
- Create a GitHub release

## 🔧 Advanced Release

For more control, use the **Release** workflow directly:

1. Go to **Actions** → **Release**
2. Click **Run workflow**
3. Configure options:
   - **Version**: Manual version (e.g., "1.2.3") or leave empty for auto-bump
   - **Version Type**: patch/minor/major (ignored if manual version provided)
   - **Create Tag**: Whether to create and push git tag
   - **Skip Build**: Only bump version without building/releasing

## 📦 Supported Platforms

The workflow builds wheels for:

### Linux (glibc)
- x86_64, x86, aarch64, armv7, s390x, ppc64le

### Linux (musl)
- x86_64, x86, aarch64, armv7

### Windows
- x64, x86

### macOS
- x86_64 (Intel)
- aarch64 (Apple Silicon)

## 🔑 Requirements

- `PYPI_API_TOKEN` secret must be configured in repository settings
- Workflows require `contents: write` permissions for version commits and tags

## 🏷️ Tag-based Releases

You can also trigger releases by pushing tags:

```bash
git tag v1.2.3
git push origin v1.2.3
```

This will automatically build and publish without version bumping.

## 🔍 Monitoring

- Check the **Actions** tab for workflow status
- Release summaries show built wheels and installation commands
- Failed builds will show detailed logs for debugging

## 🛠️ Troubleshooting

### Build Failures
- Check individual platform build logs
- Ensure Rust code compiles on all targets
- Verify Python bindings are compatible

### PyPI Upload Issues
- Verify `PYPI_API_TOKEN` is valid
- Check if version already exists (workflow skips existing)
- Ensure package name matches PyPI project

### Version Conflicts
- Workflow validates version format (x.y.z)
- Manual versions override automatic bumping
- Git conflicts require manual resolution