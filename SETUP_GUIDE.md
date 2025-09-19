# BM25-RS Setup Guide for PyPI Publishing

This guide walks you through setting up and publishing the BM25-RS package to PyPI.

## Prerequisites

### 1. Development Environment
- **Python 3.8+**: Install from [python.org](https://www.python.org/downloads/)
- **Rust**: Install from [rustup.rs](https://rustup.rs/)
- **Git**: For version control

### 2. Required Tools
```bash
# Install maturin for building Python extensions from Rust
pip install maturin

# Install development dependencies
pip install pytest black ruff mypy twine

# Install optional dependencies for benchmarking
pip install psutil matplotlib numpy
```

## Project Structure

```
bm25-rs/
├── src/                    # Rust source code
│   ├── lib.rs             # Main library entry point
│   ├── bm25okapi.rs       # BM25Okapi implementation
│   ├── bm25plus.rs        # BM25Plus implementation
│   ├── bm25l.rs           # BM25L implementation
│   └── optimizations.rs   # Performance optimizations
├── python/bm25_rs/        # Python package
│   ├── __init__.py        # Package initialization
│   ├── utils.py           # Utility functions
│   └── benchmarks.py      # Benchmarking tools
├── tests/                 # Test suite
├── examples/              # Usage examples
├── scripts/               # Build and release scripts
├── .github/workflows/     # CI/CD configuration
├── Cargo.toml            # Rust package configuration
├── pyproject.toml        # Python package configuration
├── README.md             # Package documentation
├── LICENSE               # MIT license
└── CHANGELOG.md          # Version history
```

## Development Workflow

### 1. Local Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/bm25-rs.git
cd bm25-rs

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
maturin develop --release

# Run tests
pytest tests/ -v

# Run examples
python examples/basic_usage.py
python examples/performance_demo.py
```

### 2. Code Quality Checks

```bash
# Format Rust code
cargo fmt

# Lint Rust code
cargo clippy -- -D warnings

# Format Python code
black python/ examples/ tests/ scripts/

# Lint Python code
ruff check python/ examples/ tests/ scripts/

# Type check Python code
mypy python/bm25_rs/ --ignore-missing-imports
```

### 3. Building and Testing

```bash
# Build development version
python scripts/build.py --dev

# Build wheel and source distribution
python scripts/build.py --wheel --sdist

# Run comprehensive tests
python scripts/build.py --test

# Run benchmarks
python scripts/build.py --benchmark

# Build everything
python scripts/build.py --all
```

## PyPI Publishing Setup

### 1. Create PyPI Accounts
- **PyPI**: [pypi.org/account/register](https://pypi.org/account/register/)
- **Test PyPI**: [test.pypi.org/account/register](https://test.pypi.org/account/register/)

### 2. Configure API Tokens

Create API tokens for secure uploading:

1. Go to PyPI Account Settings → API tokens
2. Create a new token with appropriate scope
3. Configure in your environment:

```bash
# Option 1: Environment variables
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-api-token-here

# Option 2: .pypirc file
cat > ~/.pypirc << EOF
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-token-here
EOF
```

### 3. Update Package Metadata

Edit `pyproject.toml` to update:
- Package name (ensure it's unique on PyPI)
- Author information
- Repository URLs
- Version number

Edit `Cargo.toml` to update:
- Package metadata
- Author information
- Repository URLs

## Release Process

### 1. Prepare Release

```bash
# Update version in pyproject.toml and Cargo.toml
# Update CHANGELOG.md with new features/fixes
# Commit all changes
git add .
git commit -m "Prepare release v0.1.0"
```

### 2. Test Release (Recommended)

```bash
# Test on Test PyPI first
python scripts/release.py --test

# Install and test the package
pip install --index-url https://test.pypi.org/simple/ bm25-rs==0.1.0

# Test basic functionality
python -c "from bm25_rs import BM25Okapi; print('Import successful')"
```

### 3. Production Release

```bash
# Release to PyPI
python scripts/release.py

# Or step by step:
python scripts/release.py --dry-run  # Test without uploading
python scripts/release.py            # Actual release
```

### 4. Verify Release

```bash
# Install from PyPI
pip install bm25-rs

# Test installation
python -c "
from bm25_rs import BM25Okapi, BM25Plus, BM25L
print('BM25-RS installed successfully!')
print(f'Available classes: BM25Okapi, BM25Plus, BM25L')
"
```

## Continuous Integration

The project includes GitHub Actions workflows for:

### CI Pipeline (`.github/workflows/ci.yml`)
- **Testing**: Runs on Python 3.8-3.12, Linux/Windows/macOS
- **Linting**: Code formatting and style checks
- **Building**: Creates wheels for all platforms

### Release Pipeline (`.github/workflows/release.yml`)
- **Triggered**: On git tags matching `v*`
- **Builds**: Wheels for all platforms + source distribution
- **Publishes**: Automatically to PyPI using trusted publishing
- **Creates**: GitHub release with changelog

### Setting Up GitHub Actions

1. **Enable Actions**: In your GitHub repository settings
2. **Configure Secrets**: For PyPI publishing (if not using trusted publishing)
3. **Set up Trusted Publishing**: Recommended for security
   - Go to PyPI → Manage → Publishing
   - Add GitHub repository as trusted publisher

## Package Maintenance

### Version Management
- Follow [Semantic Versioning](https://semver.org/)
- Update version in both `pyproject.toml` and `Cargo.toml`
- Update `CHANGELOG.md` with each release

### Performance Monitoring
```bash
# Run benchmarks regularly
python python/bm25_rs/benchmarks.py

# Profile memory usage
python examples/performance_demo.py
```

### Security Updates
- Monitor dependencies for security issues
- Update Rust and Python dependencies regularly
- Run security audits:

```bash
# Rust security audit
cargo audit

# Python security audit
pip-audit
```

## Troubleshooting

### Common Build Issues

1. **Rust not found**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Maturin build fails**:
   ```bash
   pip install --upgrade maturin
   cargo clean
   maturin develop --release
   ```

3. **Import errors**:
   ```bash
   # Ensure you're in the right environment
   which python
   pip list | grep bm25
   ```

### Platform-Specific Issues

**Windows**:
- Install Visual Studio Build Tools
- Use PowerShell or Command Prompt
- May need to set `RUSTFLAGS="-C target-feature=+crt-static"`

**macOS**:
- Install Xcode Command Line Tools: `xcode-select --install`
- For Apple Silicon, ensure Rust targets are installed

**Linux**:
- Install build essentials: `sudo apt-get install build-essential`
- May need additional system libraries

## Support and Contributing

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/yourusername/bm25-rs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/bm25-rs/discussions)
- **Documentation**: [README.md](README.md)

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

### Development Guidelines
- Write tests for new features
- Maintain backward compatibility
- Update documentation
- Follow code style guidelines
- Add benchmarks for performance changes

---

This setup guide should help you successfully publish and maintain the BM25-RS package on PyPI. For questions or issues, please refer to the project documentation or create an issue on GitHub.