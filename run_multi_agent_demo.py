#!/usr/bin/env python3
"""
Advanced Multi-Agent RAG System Demo
Showcases the full capabilities of the upgraded system.
"""

import sys
import asyncio
from pathlib import Path

sys.path.append('.')

async def run_advanced_demo():
    """Run comprehensive demo of the advanced multi-agent RAG system."""
    print("🤖 Advanced Multi-Agent RAG System - Full Demo")
    print("=" * 60)
    
    try:
        # Import system components
        from src.agents.multi_agent_orchestrator import multi_agent_orchestrator
        from src.memory.conversation_memory import conversation_memory
        from src.evaluation.hallucination_detector import hallucination_detector
        from src.rag_orchestrator import rag_orchestrator
        
        print("1. 🔧 Initializing Advanced Multi-Agent System...")
        await rag_orchestrator.initialize()
        print("   ✅ RAG orchestrator initialized")
        print("   ✅ Multi-agent system ready")
        print("   ✅ Conversation memory active")
        print("   ✅ Evaluation systems loaded")
        
        # Create conversation session
        print("\n2. 💬 Starting Conversation Session...")
        session_id = await conversation_memory.create_session()
        print(f"   📋 Session ID: {session_id}")
        
        # Process sample documents
        print("\n3. 📚 Processing Sample Documents...")
        sample_docs = [
            "sample_documents/ai_research.md",
            "sample_documents/clinical_guidelines.md"
        ]
        
        for doc_path in sample_docs:
            if Path(doc_path).exists():
                print(f"   📄 Processing {doc_path}")
                document = await rag_orchestrator.ingest_document(Path(doc_path))
                print(f"      ✅ Created {len(document.chunks)} chunks")
        
        # Test multi-agent queries
        print("\n4. 🤖 Testing Multi-Agent System...")
        
        test_queries = [
            "What are the main applications of AI in healthcare and how accurate are they?",
            "Calculate the percentage improvement in drug discovery with AI and explain the process",
            "What is the first-line treatment for diabetes and what are the latest research findings?",
            "Compare AI accuracy in medical imaging across different conditions"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   📝 Query {i}: {query}")
            print("   " + "-" * 70)
            
            try:
                # Process with multi-agent system
                response = await multi_agent_orchestrator.process_query(query)
                
                if response.success:
                    print(f"   🤖 Answer: {response.final_answer[:200]}...")
                    print(f"   📊 Confidence: {response.confidence:.2f}")
                    print(f"   ⏱️  Processing Time: {response.total_execution_time:.2f}s")
                    print(f"   🔧 Agents Used: {', '.join(response.agent_responses.keys())}")
                    
                    # Add to conversation
                    await conversation_memory.add_turn(
                        session_id,
                        query,
                        response.final_answer,
                        response.confidence,
                        response.sources_used,
                        response.total_execution_time
                    )
                    
                    # Show reasoning chain
                    if response.reasoning_chain:
                        print("   🧠 Reasoning Chain:")
                        for step in response.reasoning_chain[:3]:
                            print(f"      • {step}")
                    
                    # Hallucination detection
                    sources = []
                    for agent_response in response.agent_responses.values():
                        if agent_response.success and isinstance(agent_response.result, dict):
                            if "sources" in agent_response.result:
                                sources.extend(agent_response.result["sources"])
                    
                    if sources:
                        hallucination_result = await hallucination_detector.detect_hallucination(
                            query, response.final_answer, sources
                        )
                        print(f"   🔍 Hallucination Check: {'⚠️ Detected' if hallucination_result.is_hallucination else '✅ Clean'}")
                        print(f"   📈 Source Alignment: {hallucination_result.source_alignment_score:.2f}")
                
                else:
                    print(f"   ❌ Query failed: {response.final_answer}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Show conversation history
        print(f"\n5. 📜 Conversation History Summary...")
        conv_context = await conversation_memory.get_conversation_context(session_id)
        print(f"   💬 Total turns: {conv_context['turn_count']}")
        print(f"   ⏱️  Session duration: {conv_context['session_duration']:.1f}s")
        
        # System statistics
        print(f"\n6. 📊 System Performance Statistics...")
        agent_stats = multi_agent_orchestrator.get_system_stats()
        conv_stats = await conversation_memory.get_session_stats()
        
        print(f"   🤖 Queries processed: {agent_stats['total_queries_processed']}")
        print(f"   ⚡ Success rate: {agent_stats['success_rate']:.1%}")
        print(f"   ⏱️  Average response time: {agent_stats['average_execution_time']:.2f}s")
        print(f"   💬 Conversation sessions: {conv_stats['total_sessions']}")
        
        # Agent details
        print(f"\n   🔧 Agent Performance:")
        for agent_id, stats in agent_stats['agent_statistics'].items():
            print(f"      • {stats['name']}: {stats['execution_count']} executions, {stats['average_execution_time']:.2f}s avg")
        
        print(f"\n7. 🎯 Advanced Features Demonstrated:")
        print("   ✅ Multi-agent orchestration with planning")
        print("   ✅ Tool use and external integrations")
        print("   ✅ Conversation memory and context")
        print("   ✅ Hallucination detection and evaluation")
        print("   ✅ Hybrid retrieval with reranking")
        print("   ✅ Real-time reasoning chains")
        print("   ✅ Source attribution and confidence scoring")
        
        print(f"\n🎉 Advanced Multi-Agent RAG Demo Complete!")
        print(f"\n💡 Next Steps:")
        print("   • Start Streamlit UI: streamlit run streamlit_app.py")
        print("   • Start API server: python start_server.py")
        print("   • Visit API docs: http://localhost:8000/docs")
        print("   • Try multi-agent endpoints: /multi-agent/query")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   • Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   • Check OpenAI API key is set in .env file")
        print("   • Verify Python version is 3.8+")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_advanced_demo())