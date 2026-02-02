# Contributing to Advanced RAG System

Thank you for your interest in contributing to the Advanced RAG System! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- OpenAI API key (for testing)

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/advanced-rag-system.git
   cd advanced-rag-system
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Copy environment file:
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

## 🔧 Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add docstrings for all classes and functions
- Keep functions focused and small

### Testing
- Write tests for all new features
- Ensure all tests pass before submitting PR
- Run tests with: `pytest`
- Aim for >80% code coverage

### Documentation
- Update README.md for new features
- Add docstrings to all public methods
- Include examples for complex features

## 📝 Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. Make your changes and commit:
   ```bash
   git commit -m "Add amazing feature"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```

4. Create a Pull Request with:
   - Clear description of changes
   - Link to related issues
   - Screenshots if applicable
   - Test results

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs

## 💡 Feature Requests

For new features:
- Check existing issues first
- Describe the use case
- Explain why it would be valuable
- Consider implementation complexity

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to docs
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

## 📞 Getting Help

- Check the documentation first
- Search existing issues
- Ask questions in discussions
- Contact maintainers for complex issues

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing! 🎉