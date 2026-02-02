# Project Structure

```
advanced-multi-agent-rag/
├── .github/                    # GitHub workflows and templates
│   ├── workflows/
│   │   └── ci.yml             # CI/CD pipeline
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── docs/                      # Documentation
│   ├── API.md                 # API documentation
│   └── DEPLOYMENT.md          # Deployment guide
├── examples/                  # Usage examples
│   ├── complete_rag_examples.py
│   └── document_processing_examples.py
├── sample_documents/          # Sample documents for testing
│   ├── ai_research.md
│   └── clinical_guidelines.md
├── src/                       # Main source code
│   ├── agents/               # Multi-agent system
│   │   ├── base_agent.py
│   │   ├── planning_agent.py
│   │   ├── tool_agent.py
│   │   ├── synthesis_agent.py
│   │   └── multi_agent_orchestrator.py
│   ├── analytics/            # Analytics and monitoring
│   │   └── dashboard.py
│   ├── api/                  # FastAPI application
│   │   └── main.py
│   ├── embeddings/           # Embedding services
│   │   ├── embedding_service.py
│   │   └── fine_tuned_embeddings.py
│   ├── evaluation/           # Evaluation framework
│   │   ├── hallucination_detector.py
│   │   └── advanced_evaluation.py
│   ├── llm/                  # LLM providers
│   │   └── providers.py
│   ├── memory/               # Conversation memory
│   │   └── conversation_memory.py
│   ├── models/               # Data models
│   ├── processing/           # Document processing
│   ├── retrieval/            # Retrieval system
│   │   ├── vector_store.py
│   │   └── hybrid_retriever.py
│   ├── config.py             # Configuration
│   ├── main.py               # Main application
│   └── rag_orchestrator.py   # Core orchestrator
├── tests/                    # Test suite
│   └── test_performance.py
├── .env.example              # Environment template
├── .gitignore               # Git ignore rules
├── CHANGELOG.md             # Version history
├── CONTRIBUTING.md          # Contribution guidelines
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── LICENSE                  # MIT License
├── Makefile                 # Build automation
├── PROJECT_STRUCTURE.md     # This file
├── pytest.ini              # Pytest configuration
├── README.md                # Main documentation
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── advanced_demo.py         # Advanced demo
├── offline_demo.py          # Offline demo
├── quick_functionality_test.py  # Quick tests
├── run_multi_agent_demo.py  # Multi-agent demo
├── setup_api_key.py         # API key setup
├── simple_demo.py           # Simple demo
├── start_server.py          # Server startup
├── streamlit_app.py         # Streamlit UI
├── test_api.py              # API tests
└── test_full_system.py      # System tests
```

## Key Components

### Core System (`src/`)
- **RAG Orchestrator**: Central system coordinator
- **Multi-Agent System**: Planning, Tool, and Synthesis agents
- **Hybrid Retrieval**: Semantic + lexical + reranking
- **LLM Integration**: Multiple provider support
- **Conversation Memory**: Persistent chat history

### Advanced Features
- **Fine-tuned Embeddings**: Domain-specific embedding models
- **Advanced Evaluation**: RAGAS integration with bias detection
- **Analytics Dashboard**: Comprehensive monitoring
- **Tool Integration**: Web search, calculations, code execution

### Production Features
- **FastAPI Backend**: RESTful API with documentation
- **Streamlit UI**: User-friendly web interface
- **Docker Support**: Containerized deployment
- **CI/CD Pipeline**: Automated testing and deployment
- **Comprehensive Testing**: Unit, integration, and performance tests

### Documentation
- **API Documentation**: Complete endpoint reference
- **Deployment Guide**: Multiple deployment options
- **Examples**: Usage examples and tutorials
- **Contributing Guide**: Development guidelines