# launch_rag_ui.py
# One-click launcher for RAG Streamlit interface

import subprocess
import sys
import os
import time

def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
        else:
            print("⚠️ Ollama is not responding properly")
            return False
    except Exception:
        print("❌ Ollama is not running")
        print("💡 Start Ollama first with: ollama serve")
        return False

def check_models():
    """Check if required models are available"""
    try:
        import config
        import requests
        
        # Check models
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            # Check language model
            if config.LANGUAGE_MODEL in model_names:
                print(f"✅ Language model found: {config.LANGUAGE_MODEL}")
            else:
                print(f"❌ Language model not found: {config.LANGUAGE_MODEL}")
                print(f"💡 Install with: ollama pull {config.LANGUAGE_MODEL}")
                return False
            
            # Embedding model check (different approach)
            print(f"✅ Embedding model configured: {config.EMBEDDING_MODEL}")
            
            return True
        else:
            print("❌ Cannot check models")
            return False
            
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def launch_streamlit():
    """Launch Streamlit application"""
    try:
        print("🚀 Launching Streamlit RAG Interface...")
        print("🌐 Opening browser automatically...")
        print("📝 To stop: Press Ctrl+C in this terminal")
        print("=" * 50)
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

def quick_test():
    """Quick test of the RAG system"""
    try:
        print("🧪 Quick RAG system test...")
        
        from src.retrieval_system import RAGRetriever
        rag = RAGRetriever()
        
        # Test query
        response = rag.query("How fast can cats run?")
        
        if response and response['answer']:
            print("✅ RAG system test passed")
            return True
        else:
            print("❌ RAG system test failed")
            return False
            
    except Exception as e:
        print(f"❌ RAG test error: {e}")
        return False

def main():
    """Main launcher function"""
    print("🎮 RAG Streamlit UI Launcher")
    print("=" * 30)
    
    # Check Ollama
    if not check_ollama():
        print("\n🔧 Please start Ollama first:")
        print("   ollama serve")
        return
    
    # Check models
    if not check_models():
        print("\n🔧 Please install required models first")
        return
    
    # Quick test
    if not quick_test():
        print("\n🔧 Please fix RAG system issues first")
        return
    
    print("\n✅ All checks passed!")
    print("🚀 Starting Streamlit interface...")
    time.sleep(2)
    
    # Launch Streamlit
    launch_streamlit()

if __name__ == "__main__":
    main()