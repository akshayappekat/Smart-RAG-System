#!/usr/bin/env python3
"""Simple Streamlit frontend for Smart-RAG system."""

import streamlit as st
import requests
import json
import time

# Configure page
st.set_page_config(
    page_title="Smart-RAG System",
    page_icon="🤖",
    layout="wide"
)

# API base URL
API_BASE = "http://localhost:8000"

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🤖 Smart-RAG System")
    st.markdown("*Advanced Retrieval-Augmented Generation with Multi-Agent Intelligence*")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Health check
        if st.button("🏥 Check Health"):
            check_system_health()
        
        # System stats
        if st.button("📊 System Stats"):
            show_system_stats()
        
        st.header("⚙️ Query Settings")
        max_chunks = st.slider("Max Sources", 1, 10, 5)
        
        st.header("📖 About")
        st.markdown("""
        **Features:**
        - 🔍 Intelligent document search
        - 🤖 AI-powered answer generation  
        - 📚 Source attribution
        - ⚡ Real-time processing
        - 🏥 Health monitoring
        """)
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"**{i}. {source.get('title', 'Unknown')}**")
                            st.write(f"Score: {source.get('score', 0):.2f}")
                            st.write(f"Content: {source.get('content', '')[:200]}...")
                            st.write("---")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about healthcare, AI, or diabetes..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response_data = query_api(prompt, max_chunks)
                
                if response_data:
                    st.markdown(response_data["answer"])
                    
                    # Show confidence and timing
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"🎯 Confidence: {response_data['confidence_score']:.1%}")
                    with col2:
                        st.caption(f"⏱️ Time: {response_data['processing_time']:.2f}s")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_data["answer"],
                        "sources": response_data.get("sources", [])
                    })
                else:
                    error_msg = "❌ Sorry, I couldn't process your request. Please check if the API server is running."
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Example queries
    st.header("💡 Try These Examples")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏥 AI in Healthcare"):
            example_query("What are the main applications of AI in healthcare?")
    
    with col2:
        if st.button("💊 Diabetes Treatment"):
            example_query("What is the first-line treatment for diabetes?")
    
    with col3:
        if st.button("🔬 AI Accuracy"):
            example_query("How accurate are AI diagnostic systems?")


def query_api(query: str, max_chunks: int = 5):
    """Query the RAG API."""
    try:
        response = requests.post(
            f"{API_BASE}/query",
            json={"query": query, "max_chunks": max_chunks},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server. Make sure it's running on port 8000.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None


def check_system_health():
    """Check system health."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            st.success("✅ System is healthy!")
            
            st.subheader("Component Status:")
            for component, status in health_data["components"].items():
                icon = "✅" if status == "operational" else "❌"
                st.write(f"{icon} **{component.replace('_', ' ').title()}**: {status}")
        else:
            st.error("❌ Health check failed")
            
    except Exception as e:
        st.error(f"❌ Cannot reach API server: {e}")


def show_system_stats():
    """Show system statistics."""
    try:
        response = requests.get(f"{API_BASE}/stats", timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            
            st.subheader("📊 System Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Documents", stats.get("documents", 0))
                st.metric("🔧 System Mode", stats.get("system_mode", "unknown"))
            
            with col2:
                st.metric("📝 Total Chunks", stats.get("total_chunks", 0))
            
            if "capabilities" in stats:
                st.subheader("🚀 Capabilities")
                for capability in stats["capabilities"]:
                    st.write(f"• {capability}")
        else:
            st.error("❌ Failed to get system stats")
            
    except Exception as e:
        st.error(f"❌ Cannot reach API server: {e}")


def example_query(query: str):
    """Add example query to chat."""
    st.session_state.messages.append({"role": "user", "content": query})
    st.rerun()


if __name__ == "__main__":
    main()