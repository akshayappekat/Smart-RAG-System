#!/usr/bin/env python3
"""
Script to set up OpenAI API key for the Advanced RAG system.
Run this script to configure your API key securely.
"""

import os
from pathlib import Path

def setup_openai_key():
    """Interactive setup for OpenAI API key."""
    print("🔑 OpenAI API Key Setup")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    
    print("\nTo get your OpenAI API key:")
    print("1. Go to https://platform.openai.com/api-keys")
    print("2. Create a new API key")
    print("3. Copy the key (starts with 'sk-')")
    print()
    
    api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: OpenAI API keys typically start with 'sk-'")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != 'y':
            return
    
    # Update .env file
    env_content = f"""# OpenAI Configuration
OPENAI_API_KEY={api_key}

# Model Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# System Configuration
LOG_LEVEL=INFO
API_PORT=8000
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ API key saved to {env_file}")
    print("\n🚀 You can now run the advanced demo:")
    print("   python advanced_demo.py")
    print("\n🌐 Or start the API server:")
    print("   python -m uvicorn src.api.main:app --reload")

if __name__ == "__main__":
    setup_openai_key()