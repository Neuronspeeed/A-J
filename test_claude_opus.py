#!/usr/bin/env python3
"""
Quick test to verify Claude 4 Opus configuration and setup.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_claude_opus_config():
    """Test Claude 4 Opus configuration."""
    try:
        from config.experiments import PHASE1_CONFIG, get_provider_config
        
        print("🧪 Testing Claude 4 Opus Configuration")
        print("=" * 50)
        
        # Check if Claude 4 Opus is in the model list
        models = PHASE1_CONFIG.model_names
        print(f"📋 Available models: {models}")
        
        if "claude-4-opus-20250514" in models:
            print("✅ Claude 4 Opus found in model list")
        else:
            print("❌ Claude 4 Opus NOT found in model list")
            return False
        
        # Test provider configuration
        try:
            config = get_provider_config("claude-4-opus-20250514")
            print(f"✅ Provider config found:")
            print(f"   Model: {config.model_name}")
            print(f"   Provider: {config.provider_type}")
            print(f"   API Key Var: {config.api_key_env_var}")
            print(f"   Max Tokens: {config.max_tokens}")
        except Exception as e:
            print(f"❌ Provider config error: {e}")
            return False
        
        # Test provider creation
        try:
            from core.llm_providers import create_provider
            provider = create_provider(config)
            print(f"✅ Provider created: {type(provider).__name__}")
        except Exception as e:
            print(f"❌ Provider creation error: {e}")
            return False
        
        print("\n🎉 All tests passed! Claude 4 Opus is ready.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_claude_opus_config()
    sys.exit(0 if success else 1)
