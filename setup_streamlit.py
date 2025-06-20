# setup_streamlit.py
# Setup script for Streamlit RAG interface

import subprocess
import sys
import os

def install_streamlit():
    """Install Streamlit if not already installed"""
    try:
        import streamlit
        print("✅ Streamlit already installed")
        return True
    except ImportError:
        print("📦 Installing Streamlit...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit")
            return False

def check_rag_system():
    """Check if RAG system is ready"""
    try:
        # Check if required files exist
        required_files = [
            "config.py",
            "data/cat-facts.txt",
            "src/retrieval_system.py"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print("❌ Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            return False
        
        print("✅ All required files found")
        return True
        
    except Exception as e:
        print(f"❌ Error checking RAG system: {e}")
        return False

def create_requirements_txt():
    """Create requirements.txt for the project"""
    requirements = [
        "streamlit>=1.28.0",
        "numpy",
        "requests",
        "ollama"
    ]
    
    try:
        with open("requirements.txt", "w") as f:
            for req in requirements:
                f.write(f"{req}\n")
        print("✅ Created requirements.txt")
    except Exception as e:
        print(f"❌ Error creating requirements.txt: {e}")

def main():
    """Main setup function"""
    print("🚀 Setting up Streamlit RAG Interface")
    print("=" * 40)
    
    # Install Streamlit
    if not install_streamlit():
        return False
    
    # Check RAG system
    if not check_rag_system():
        print("\n🔧 Setup steps needed:")
        print("1. Make sure config.py exists")
        print("2. Run: python download_data.py")
        print("3. Test RAG system first")
        return False
    
    # Create requirements.txt
    create_requirements_txt()
    
    print("\n🎉 Setup complete!")
    print("\n🚀 To start the Streamlit app:")
    print("   streamlit run streamlit_app.py")
    print("\n🌐 It will open in your browser automatically!")
    
    return True

if __name__ == "__main__":
    main()