# deploy_setup.py
# Deployment setup and verification script

import os
import subprocess
import sys

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv/

# Data files
*.npy
*.pkl
rag_vector_database/
test_*.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Environment variables
.env
.env.local

# Streamlit
.streamlit/secrets.toml
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("✅ Created .gitignore")

def check_files():
    """Check if all required files exist"""
    required_files = [
        'streamlit_app.py',
        'config.py',
        'download_data.py',
        'requirements.txt',
        'src/retrieval_system.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ All required files present")
        return True

def setup_github():
    """Setup Git repository"""
    try:
        # Initialize git if not already done
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True)
            print("✅ Git repository initialized")
        else:
            print("✅ Git repository already exists")
        
        # Create .gitignore
        create_gitignore()
        
        # Add files
        subprocess.run(['git', 'add', '.'], check=True)
        print("✅ Files staged for commit")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git setup error: {e}")
        return False
    except FileNotFoundError:
        print("❌ Git not installed. Please install Git first.")
        return False

def create_render_button():
    """Create render deploy button link"""
    repo_url = "https://github.com/yourusername/rag-chat-assistant"  # Update this
    render_url = f"https://render.com/deploy?repo={repo_url}"
    
    print(f"\n🚀 Render Deploy Button:")
    print(f"Add this to your README.md:")
    print(f'[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)]({render_url})')

def deployment_checklist():
    """Show deployment checklist"""
    print("\n📋 Deployment Checklist:")
    print("=" * 40)
    
    checklist = [
        "✅ All files present",
        "✅ Git repository set up",
        "✅ .gitignore created",
        "🔄 Push to GitHub",
        "🔄 Create Render account",
        "🔄 Connect GitHub to Render",
        "🔄 Set environment variables (if needed)",
        "🔄 Deploy!"
    ]
    
    for item in checklist:
        print(f"  {item}")
    
    print(f"\n🌐 GitHub Steps:")
    print(f"1. Create repository on GitHub")
    print(f"2. git remote add origin https://github.com/yourusername/repo-name.git")
    print(f"3. git commit -m 'Initial commit'")
    print(f"4. git push -u origin main")
    
    print(f"\n☁️ Render Steps:")
    print(f"1. Go to render.com")
    print(f"2. Connect GitHub account")
    print(f"3. Create new Web Service")
    print(f"4. Select your repository")
    print(f"5. Deploy!")

def main():
    """Main deployment setup"""
    print("🚀 RAG Project Deployment Setup")
    print("=" * 35)
    
    # Check files
    if not check_files():
        print("\n🔧 Please create missing files first")
        return
    
    # Setup Git
    if setup_github():
        print("✅ Git setup complete")
    else:
        print("❌ Git setup failed")
        return
    
    # Show deployment info
    create_render_button()
    deployment_checklist()
    
    print(f"\n🎉 Your RAG project is ready for deployment!")
    print(f"📚 Next: Push to GitHub and deploy to Render")

if __name__ == "__main__":
    main()