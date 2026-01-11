# SOLVRO MCP - Knowledge Graph RAG System

![MCP Server Diagram](docs/images/logo.png)

A production-ready Model Context Protocol (MCP) server implementing a Retrieval-Augmented Generation (RAG) system with Neo4j graph database backend. The system intelligently converts natural language queries into Cypher queries, retrieves relevant information from a knowledge graph, and provides contextual answers about Wroclaw University of Science and Technology.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development](#development)
- [API Reference](#api-reference)
- [Observability](#observability)
- [Contributing](#contributing)

## Architecture Overview

The system consists of three main components:

1. **MCP Server** - FastMCP-based server exposing knowledge graph tools
2. **MCP Client** - CLI interface for querying the knowledge graph
3. **Data Pipeline** - Multi-threaded ETL pipeline for loading documents into Neo4j

### Data Flow
![Architecture Diagram](docs/images/kg_pipeline.png)
```
User Query → MCP Client → MCP Server → RAG System → Neo4j Graph DB
                                    ↓
                            Langfuse Observability
                                    ↓
                            LLM Processing → Response
```


### Key Technologies

- **FastMCP**: Model Context Protocol server implementation
- **LangChain**: LLM orchestration and chaining
- **LangGraph**: State machine for RAG pipeline
- **Neo4j**: Graph database for knowledge storage
- **Langfuse**: Observability and tracing
- **OpenAI/DeepSeek**: LLM providers for query generation and answering

## Features

### Intelligent Query Routing

- **Guardrails System**: Automatically determines if queries are relevant to the knowledge base
- **Fallback Mechanism**: Returns graceful responses for out-of-scope queries
- **Session Tracking**: Full trace tracking across the entire query lifecycle

### Advanced RAG Pipeline

- **Multi-Stage Processing**: Guardrails → Cypher Generation → Retrieval → Response
- **State Machine Architecture**: Built with LangGraph for predictable execution flow
- **Error Handling**: Robust error recovery with fallback strategies
- **Caching**: Schema caching for improved performance

### Dual LLM Strategy

- **Fast LLM (gpt-5-nano)**: Quick decision-making for guardrails
- **Accurate LLM (gpt-5-mini)**: Precise Cypher query generation

### Observability

- **Langfuse Integration**: Complete trace and session tracking
- **Mermaid Visualization**: Graph flow visualization for debugging
- **Structured Logging**: Comprehensive logging throughout the pipeline

### Data Pipeline

- **Multi-threaded Processing**: Configurable thread pool for parallel document processing
- **PDF Support**: Extract and process PDF documents
- **Dynamic Graph Schema**: Configurable nodes and relationships via yaml
- **Database Management**: Built-in database clearing and initialization

**Data Pipeline Diagram**

 - **Diagram:** Displays the pipeline flow for document ingestion and processing. See the image below:

  ![Data Pipeline](docs/images/data_pipeline.png)


## Prerequisites

- Python 3.12 or higher
- Neo4j database instance (local or cloud)
- OpenAI API key or DeepSeek API key
- Langfuse account (for observability)
- `uv` package manager (recommended) or `pip`

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SOLVRO_MCP
```

### 2. Install Dependencies

Using `uv` (recommended):

```bash
uv sync
```

Using `pip`:

```bash
pip install -e .
```

### 3. Set Up Neo4j

**Option A: Docker Compose**

```bash
cd db
docker-compose up -d
```

**Option B: Neo4j Desktop or Aura**

Follow Neo4j's installation guide for your platform.

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# LLM Provider (choose one)
OPENAI_API_KEY=your_openai_api_key
# OR
DEEPSEEK_API_KEY=your_deepseek_api_key

# CLARIN Polish LLM (optional, for Polish language support)
CLARIN_API_KEY=your_clarin_api_key

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Langfuse Observability
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Configuration

### Centralized Configuration (`graph_config.yaml`)

All application settings are centralized in `graph_config.yaml`, including:

- **Server Configuration**: MCP server and ToPWR API ports, hosts, and transport
- **LLM Models**: Model names and temperatures for fast and accurate operations
- **RAG Settings**: Max results, debug mode
- **Data Pipeline**: Chunk sizes, overlaps, token limits
- **Neo4j Database**: Connection settings (with environment variable support)
- **Observability**: Langfuse configuration
- **Graph Schema**: Node types and relationship types
- **Prompts**: All prompt templates used throughout the system

#### Example Configuration

```yaml
servers:
  mcp:
    transport: "http"
    port: 8005
    host: "127.0.0.1"
  topwr_api:
    host: "0.0.0.0"
    port: 8000
    cors_origins: "*"

llm:
  fast_model:
    name: "gpt-5-nano"
    temperature: 0.1
  accurate_model:
    name: "gpt-5-mini"
    temperature: 0

rag:
  max_results: 5
  enable_debug: false

data_pipeline:
  max_chunk_size: 30000
  chunk_overlap: 200
  token_limit: 65536

nodes:
  - Document
  - Article
  - Student
  - Course
  # ... more node types

relations:
  - teaches
  - enrolls_in
  - attends
  # ... more relationship types

prompts:
  - name: final_answer
    template: |
      Your prompt template here...
```

#### Using Configuration in Code

The configuration is accessed via the `config` module:

```python
from config import get_config, get_prompt

# Get configuration instance
config = get_config()

# Access nested values with defaults
model_name = config.get_nested('llm', 'fast_model', 'name', default='gpt-5-nano')
max_results = config.get_nested('rag', 'max_results', default=5)

# Get formatted prompts with variable injection
prompt = get_prompt(
    "final_answer",
    user_input="What is AI?",
    data={"answer": "Artificial Intelligence"}
)
```

#### Environment Variables

Sensitive credentials are still managed via environment variables (`.env` file):

```yaml
neo4j:
  uri: ${NEO4J_URI}
  username: ${NEO4J_USER}
  password: ${NEO4J_PASSWORD}
```

## Usage

### Running the MCP Server

Start the FastMCP server on port 8005:

```bash
just mcp-server
# OR
uv run server
```

The server will initialize the RAG system and expose the `knowledge_graph_tool`.

### Querying via CLI Client

Query the knowledge graph using natural language:

```bash
just mcp-client "Czym jest nagroda dziekana?"
# OR
uv run kg "What is the dean's award?"
```

Example queries:

```bash
# Polish queries
uv run kg "Jakie są wymagania dla stypendium rektora?"
uv run kg "Kiedy są terminy egzaminów?"

# English queries
uv run kg "What are the scholarship requirements?"
uv run kg "When are the exam dates?"
```

### Data Pipeline

#### Load Documents into Neo4j

```bash
just pipeline
# OR
uv run pipeline
```

#### Clear Database and Reload

```bash
just pipeline-clear
# OR
uv run pipeline --clear-db
```

#### Manual Pipeline Execution

```bash
uv run python src/scripts/data_pipeline/main.py \
  data/ \
  graph_config.yaml \
  4 \
  --clear-db
```

Parameters:
- `data/` - Input directory containing PDF files
- `graph_config.yaml` - Graph schema configuration (YAML format)
- `4` - Number of parallel threads
- `--clear-db` - (Optional) Clear database before loading

## Project Structure

```
SOLVRO_MCP/
├── src/
│   ├── mcp_server/          # MCP server implementation
│   │   ├── server.py        # FastMCP server entry point
│   │   └── tools/
│   │       └── knowledge_graph/
│   │           ├── rag.py           # RAG system core logic
│   │           ├── state.py         # LangGraph state definitions
│   │           └── graph_visualizer.py  # Mermaid visualization
│   │
│   ├── mcp_client/          # CLI client
│   │   └── client.py        # Client implementation with Langfuse integration
│   │
│   └── scripts/
│       └── data_pipeline/   # ETL pipeline
│           ├── main.py      # Pipeline orchestrator
│           ├── data_pipe.py # Data processing logic
│           ├── llm_pipe.py  # LLM-based entity extraction
│           └── pdf_loader.py # PDF document loader
│
├── db/
│   └── docker-compose.yaml  # Neo4j container configuration
│
├── data/                    # Input documents directory
├── graph_config.yaml        # Centralized configuration file
├── pyproject.toml          # Project dependencies and metadata
├── justfile                # Task runner configuration
└── README.md               # This file
```

## Development

### Code Quality

The project uses Ruff for linting and formatting:

```bash
just lint
# OR
uv run ruff format src
uv run ruff check src
```

### Configuration

**Ruff Settings** (in `pyproject.toml`):
- Line length: 100 characters
- Target: Python 3.13
- Selected rules: E, F, I, N, W

### Adding New Tools

To add a new MCP tool:

1. Create a new function in `src/mcp_server/server.py`
2. Decorate with `@mcp.tool`
3. Add documentation and type hints
4. Update the README

Example:

```python
@mcp.tool
async def new_tool(param: str) -> str:
    """
    Tool description.
    
    Args:
        param: Parameter description
    
    Returns:
        Result description
    """
    # Implementation
    return result
```

## API Reference

### MCP Server

#### `knowledge_graph_tool`

Query the knowledge graph with natural language.

**Parameters:**
- `user_input` (str): User's question or query
- `trace_id` (str, optional): Trace ID for observability

**Returns:**
- `str`: JSON string containing retrieved context or "W bazie danych nie ma informacji"

**Example:**

```python
result = await client.call_tool(
    "knowledge_graph_tool",
    {
        "user_input": "What are the scholarship requirements?",
        "trace_id": "unique-trace-id"
    }
)
```

### RAG System

#### `RAG.ainvoke()`

Async method to query the RAG system.

**Parameters:**
- `message` (str): User's question
- `session_id` (str): Session identifier (default: "default")
- `trace_id` (str): Trace identifier (default: "default")
- `callback_handler` (CallbackHandler): Langfuse callback handler

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `answer` (str): JSON context or "W bazie danych nie ma informacji"
  - `metadata` (dict):
    - `guardrail_decision` (str): Routing decision
    - `cypher_query` (str): Generated Cypher query
    - `context` (list): Retrieved data from Neo4j

### Data Pipeline

#### `DataPipe.load_data_from_directory()`

Load and process PDF documents from a directory.

**Parameters:**
- `directory` (str): Path to directory containing PDF files

**Returns:**
- None (processes documents in-place)

## Observability

### Langfuse Integration

The system provides comprehensive observability through Langfuse:

1. **Session Tracking**: All queries within a session are grouped together
2. **Trace Hierarchy**: Multi-level traces showing:
   - Guardrails decision
   - Cypher generation
   - Neo4j retrieval
   - Final answer generation
3. **Metadata Tagging**: Traces tagged with component identifiers
4. **Performance Metrics**: Latency and token usage tracking

### Viewing Traces

1. Log in to your Langfuse dashboard
2. Navigate to "Sessions" to see grouped queries
3. Click on individual traces for detailed execution flow
4. Use filters to search by tags: `mcp_client`, `knowledge_graph`, `guardrails`, etc.

### Graph Visualization

The RAG system includes Mermaid graph visualization:

```python
print(rag.visualizer.draw_mermaid())
```

This outputs a Mermaid diagram showing the state machine flow.

## Error Handling

### Common Issues

**1. Connection Refused (Neo4j)**

```
Error: Could not connect to Neo4j at bolt://localhost:7687
```

**Solution:** Ensure Neo4j is running:
```bash
cd db && docker-compose up -d
```

**2. API Key Issues**

```
Error: Missing required environment variables
```

**Solution:** Check your `.env` file contains all required keys.

**3. Import Errors**

```
ImportError: cannot import name 'langfuse_context'
```

**Solution:** This import is not available in standard Langfuse. Use session tracking through function parameters.

## Performance Tuning

### Thread Configuration

Adjust parallel processing threads in the pipeline:

```bash
uv run python src/scripts/data_pipeline/main.py data/ graph_config.yaml 8
```

Recommended thread counts:
- CPU-bound: Number of CPU cores
- I/O-bound: 2-4x CPU cores

### Neo4j Optimization

1. **Indexing**: Create indexes on frequently queried properties
2. **LIMIT Clauses**: Pipeline automatically adds LIMIT to queries
3. **Connection Pooling**: FastMCP handles connection management

### LLM Configuration

LLM settings are configured in `graph_config.yaml`:

```yaml
llm:
  fast_model:
    name: "gpt-5-nano"       # Quick decisions (guardrails)
    temperature: 0.1          # Slightly creative
  accurate_model:
    name: "gpt-5-mini"        # Precise Cypher generation  
    temperature: 0            # Fully deterministic
```

To use different models or adjust parameters, edit the YAML file - no code changes needed.
## AI Coding Guidelines

For AI coding assistants and developers, see [.github/agents.md](.github/agents.md) for detailed coding guidelines and patterns.

## ToPWR Integration API

The project now includes a FastAPI service for integrating with the ToPWR application. This provides:

- RESTful API endpoints for chat functionality
- Conversation state management with memory
- User tracking and session management
- Thread-safe in-memory storage (database integration pending)

### Running the ToPWR API

```bash
# Start the API server
just topwr-api
# OR
uv run topwr-api
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `POST /api/chat` - Send a message and get AI response
- `GET /api/sessions/{session_id}` - Get session information
- `GET /api/sessions/{session_id}/history` - Get conversation history
- `GET /api/users/{user_id}/sessions` - Get all user sessions
- `GET /api/stats` - Get system statistics
- `GET /health` - Health check

### Example API Usage

```bash
# Start a new conversation
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "message": "Czym jest nagroda dziekana?"
  }'

# Continue conversation with session_id
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "session_id": "abc123...",
    "message": "A jakie są wymagania?"
  }'
```

For complete API documentation, see [src/topwr_api/README.md](src/topwr_api/README.md)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and ensure tests pass
4. Run linting: `just lint`
5. Commit changes: `git commit -m "Add new feature"`
6. Push to branch: `git push origin feature/new-feature`
7. Open a Pull Request

## License

[Add your license here]

## Acknowledgments

- Built for Wroclaw University of Science and Technology
- Powered by FastMCP, LangChain, and Neo4j
- Observability by Langfuse

## Support

For issues and questions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation at [link]
