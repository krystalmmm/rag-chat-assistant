# streamlit_app.py
# Streamlit RAG Chat Interface

import streamlit as st
import sys
import os
import time
from datetime import datetime

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import your modules
try:
    import config
    from src.retrieval_system import RAGRetriever
    print("✅ Successfully imported RAG components")
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 RAG Chat Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    .bot-message {
        background-color: #e8f4f8;
        border-left-color: #32cd32;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #6c757d;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system (cached for performance)"""
    try:
        with st.spinner("🔧 Initializing RAG system..."):
            rag = RAGRetriever()
            
            # Check if documents exist
            stats = rag.get_system_stats()
            if stats['database']['total_documents'] == 0:
                st.warning("📚 No documents found. Adding cat facts...")
                success = rag.add_documents_from_file("cat-facts.txt")
                if not success:
                    st.error("❌ Failed to load documents. Please ensure cat-facts.txt exists.")
                    return None
            
            return rag
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {e}")
        return None

def display_chat_message(role, content, sources=None):
    """Display a chat message with styling"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🤔 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources if provided
        if sources and len(sources) > 0:
            with st.expander(f"📚 Sources ({len(sources)} found)", expanded=False):
                for i, source in enumerate(sources):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i+1}</strong> (Similarity: {source['similarity']:.3f})<br>
                        <em>{source['text']}</em>
                    </div>
                    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Chat Assistant</h1>
        <p>Ask questions about cats! Powered by your RAG system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    
    if not rag_system:
        st.error("❌ Cannot start without RAG system. Please check your setup.")
        st.stop()
    
    # Sidebar with system info
    with st.sidebar:
        st.header("🔧 System Info")
        
        # Get system stats
        if st.button("🔄 Refresh Stats"):
            st.cache_resource.clear()
        
        try:
            stats = rag_system.get_system_stats()
            st.success("✅ System Online")
            st.metric("📚 Documents", stats['database']['total_documents'])
            st.metric("💾 Database Size", f"{stats['database']['disk_usage_mb']:.1f} MB")
            st.metric("🔗 Embedding Model", config.EMBEDDING_MODEL.split('/')[-1])
            st.metric("🤖 Language Model", config.LANGUAGE_MODEL)
        except Exception as e:
            st.error(f"❌ Stats error: {e}")
        
        st.header("⚙️ Settings")
        top_k = st.slider("🔍 Number of sources", 1, 5, config.TOP_K_RETRIEVAL)
        show_timing = st.checkbox("⏱️ Show response timing", value=True)
        show_sources = st.checkbox("📚 Show sources", value=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hello! I'm your RAG assistant. Ask me anything about cats!",
            "sources": []
        })
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
                message.get("sources", []) if show_sources else None
            )
    
    # Chat input
    st.markdown("---")
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "💭 Ask your question:",
            placeholder="e.g., How fast can cats run?",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("📤 Send", type="primary")
    
    # Process user input
    if send_button and user_input.strip():
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "sources": []
        })
        
        # Show processing message
        with st.spinner("🤔 Thinking..."):
            try:
                # Get response from RAG system
                start_time = time.time()
                response = rag_system.query(user_input, top_k=top_k)
                end_time = time.time()
                
                # Prepare bot response
                bot_content = response['answer']
                
                # Add timing info if enabled
                if show_timing:
                    timing_info = f"\n\n⏱️ Response time: {response['total_time']:.2f}s"
                    bot_content += timing_info
                
                # Add bot message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_content,
                    "sources": response['sources'] if show_sources else []
                })
                
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Sorry, I encountered an error: {str(e)}",
                    "sources": []
                })
        
        # Clear input and rerun to show new messages
        st.rerun()
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("**💡 Quick Questions:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏃 How fast can cats run?"):
            st.session_state.user_input = "How fast can cats run?"
            st.rerun()
    
    with col2:
        if st.button("😴 How much do cats sleep?"):
            st.session_state.user_input = "How much do cats sleep?"
            st.rerun()
    
    with col3:
        if st.button("👂 What about cat hearing?"):
            st.session_state.user_input = "What about cat hearing?"
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Chat cleared! Ask me anything about cats!",
            "sources": []
        }]
        st.rerun()

if __name__ == "__main__":
    main()