"""Streamlit UI for the Advanced Multi-Agent RAG System."""

import streamlit as st
import asyncio
import time
import json
from typing import Dict, Any, List
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Advanced Multi-Agent RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import system components
import sys
sys.path.append('.')

from src.agents.multi_agent_orchestrator import multi_agent_orchestrator
from src.memory.conversation_memory import conversation_memory
from src.rag_orchestrator import rag_orchestrator


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False


async def initialize_system():
    """Initialize the RAG system."""
    if not st.session_state.system_initialized:
        with st.spinner("Initializing Advanced Multi-Agent RAG System..."):
            try:
                await rag_orchestrator.initialize()
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"❌ System initialization failed: {e}")
                return False
    return True


async def create_conversation_session():
    """Create new conversation session."""
    if not st.session_state.session_id:
        session_id = await conversation_memory.create_session()
        st.session_state.session_id = session_id
        st.session_state.messages = []


def main():
    """Main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🤖 Advanced Multi-Agent RAG System")
    st.markdown("*Production-grade RAG with multi-agent orchestration, tool use, and conversation memory*")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # System status
        if st.button("🚀 Initialize System"):
            asyncio.run(initialize_system())
        
        if st.button("💬 New Conversation"):
            asyncio.run(create_conversation_session())
            st.rerun()
        
        # System stats
        if st.button("📊 Show System Stats"):
            show_system_stats()
        
        # Document upload
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload documents",
            type=['pdf', 'docx', 'txt', 'md'],
            help="Upload documents to add to the knowledge base"
        )
        
        if uploaded_file and st.button("📤 Process Document"):
            process_uploaded_document(uploaded_file)
        
        # Settings
        st.header("⚙️ Settings")
        
        use_multi_agent = st.checkbox(
            "🤖 Use Multi-Agent System", 
            value=True,
            help="Enable multi-agent orchestration with planning, tools, and synthesis"
        )
        
        show_reasoning = st.checkbox(
            "🧠 Show Reasoning Chain",
            value=False,
            help="Display the reasoning process and agent interactions"
        )
        
        max_tokens = st.slider(
            "📝 Max Response Tokens",
            min_value=100,
            max_value=2000,
            value=1000,
            help="Maximum tokens for AI responses"
        )
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show additional info for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                
                if show_reasoning and "reasoning_chain" in metadata:
                    with st.expander("🧠 Reasoning Chain"):
                        for step in metadata["reasoning_chain"]:
                            st.write(f"• {step}")
                
                if "sources_used" in metadata and metadata["sources_used"]:
                    with st.expander("📚 Sources"):
                        for source in metadata["sources_used"]:
                            st.write(f"• {source}")
                
                if "execution_time" in metadata:
                    st.caption(f"⏱️ Response time: {metadata['execution_time']:.2f}s")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response_data = asyncio.run(process_query(
                    prompt, 
                    use_multi_agent, 
                    max_tokens
                ))
                
                if response_data:
                    st.markdown(response_data["content"])
                    
                    # Add assistant message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_data["content"],
                        "metadata": response_data.get("metadata", {})
                    })
                else:
                    error_msg = "❌ Sorry, I encountered an error processing your request."
                    st.markdown(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })


async def process_query(query: str, use_multi_agent: bool, max_tokens: int) -> Dict[str, Any]:
    """Process user query and return response."""
    
    try:
        # Ensure system is initialized
        if not await initialize_system():
            return None
        
        # Ensure conversation session exists
        await create_conversation_session()
        
        start_time = time.time()
        
        if use_multi_agent:
            # Use multi-agent system
            response = await multi_agent_orchestrator.process_query(query)
            
            if response.success:
                # Add to conversation memory
                await conversation_memory.add_turn(
                    st.session_state.session_id,
                    query,
                    response.final_answer,
                    response.confidence,
                    response.sources_used,
                    response.total_execution_time
                )
                
                return {
                    "content": response.final_answer,
                    "metadata": {
                        "confidence": response.confidence,
                        "execution_time": response.total_execution_time,
                        "reasoning_chain": response.reasoning_chain,
                        "sources_used": response.sources_used,
                        "agents_used": list(response.agent_responses.keys()),
                        "system_type": "multi_agent"
                    }
                }
            else:
                return {
                    "content": response.final_answer,
                    "metadata": {"system_type": "multi_agent", "error": True}
                }
        
        else:
            # Use standard RAG system
            response = await rag_orchestrator.query(query)
            execution_time = time.time() - start_time
            
            # Add to conversation memory
            await conversation_memory.add_turn(
                st.session_state.session_id,
                query,
                response.answer,
                response.confidence_score,
                [source.get("document_id", "unknown") for source in response.sources],
                execution_time
            )
            
            return {
                "content": response.answer,
                "metadata": {
                    "confidence": response.confidence_score,
                    "execution_time": execution_time,
                    "sources_used": [source.get("document_id", "unknown") for source in response.sources],
                    "system_type": "standard_rag"
                }
            }
    
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return None


def process_uploaded_document(uploaded_file):
    """Process uploaded document."""
    
    try:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save uploaded file temporarily
            import tempfile
            from pathlib import Path
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)
            
            # Process document
            async def process_doc():
                await rag_orchestrator.initialize()
                document = await rag_orchestrator.ingest_document(tmp_path)
                return document
            
            document = asyncio.run(process_doc())
            
            # Clean up temp file
            tmp_path.unlink()
            
            st.success(f"✅ Successfully processed {uploaded_file.name}")
            st.info(f"📊 Created {len(document.chunks)} chunks")
    
    except Exception as e:
        st.error(f"❌ Error processing document: {e}")


def show_system_stats():
    """Display system statistics."""
    
    try:
        # Get multi-agent stats
        agent_stats = multi_agent_orchestrator.get_system_stats()
        
        # Get conversation stats
        conv_stats = asyncio.run(conversation_memory.get_session_stats())
        
        # Get RAG stats
        rag_stats = {"status": "operational"}  # Placeholder
        
        st.subheader("📊 System Statistics")
        
        # Multi-Agent System
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🤖 Queries Processed", agent_stats["total_queries_processed"])
            st.metric("⚡ Success Rate", f"{agent_stats['success_rate']:.1%}")
        
        with col2:
            st.metric("💬 Total Sessions", conv_stats["total_sessions"])
            st.metric("🔄 Active Sessions", conv_stats["active_sessions"])
        
        with col3:
            st.metric("⏱️ Avg Response Time", f"{agent_stats['average_execution_time']:.2f}s")
            st.metric("📝 Total Turns", conv_stats["total_turns"])
        
        # Agent Details
        st.subheader("🤖 Agent Statistics")
        
        for agent_id, stats in agent_stats["agent_statistics"].items():
            with st.expander(f"Agent: {stats['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Executions:** {stats['execution_count']}")
                    st.write(f"**Avg Time:** {stats['average_execution_time']:.2f}s")
                with col2:
                    st.write(f"**Messages:** {stats['message_count']}")
                    st.write(f"**Capabilities:** {', '.join(stats['capabilities'])}")
    
    except Exception as e:
        st.error(f"Error loading system stats: {e}")


if __name__ == "__main__":
    main()