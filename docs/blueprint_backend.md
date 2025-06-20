# Backend Blueprint

## Overview

The backend is built using LangGraph, a framework for building AI agent workflows using DAGs (Directed Acyclic Graphs). It's written in Python and designed to create conversational AI agents with research capabilities.

## Project Structure

```
backend/
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── app.py           # FastAPI application
│       ├── configuration.py # Agent configuration
│       ├── graph.py        # LangGraph workflow definition
│       ├── prompts.py      # Agent prompts
│       ├── state.py        # State management
│       ├── tools_and_schemas.py # Agent tools
│       └── utils.py        # Utility functions
├── langgraph.json     # LangGraph configuration
└── pyproject.toml     # Project dependencies and settings
```

## Technology Stack

- **Python**: ^3.11
- **LangGraph**: ^0.2.6 - For building agent workflows
- **LangChain**: ^0.3.19 - For LLM tools and utilities
- **Google Gemini**: AI model integration
- **FastAPI**: Web API framework
- **Ruff**: Code formatting and linting

## Core Components

### 1. Agent Configuration

The agent is configured using environment variables and the `configuration.py` module. Key configurations include:

- Gemini API integration
- Model parameters
- Agent behavior settings

### 2. State Management

The agent uses a state management system (`state.py`) to maintain conversation context and agent memory. The state includes:

- Message history
- Research iterations
- Search queries and results

### 3. Graph Workflow

The `graph.py` defines the agent's workflow as a DAG using LangGraph. The workflow typically includes:

1. Message processing
2. Research capability
3. Response generation
4. Tool usage

### 4. Tools and Schemas

The agent has access to various tools defined in `tools_and_schemas.py`, which may include:

- Web search
- Information lookup
- Data processing
- External API integrations

### 5. API Interface

The FastAPI application (`app.py`) provides HTTP endpoints for:

- Initiating conversations
- Processing messages
- Managing agent state
- Accessing agent capabilities

## Agent Workflow

1. **Message Reception**
   - User input is received via the API
   - Messages are added to the conversation state

2. **Processing**
   - The agent processes messages through the LangGraph workflow
   - Performs research if needed (controlled by `max_research_loops`)
   - Uses available tools to gather information

3. **Response Generation**
   - Generates responses using the Gemini model
   - Incorporates research findings
   - Maintains conversation context

## State Persistence

State is managed through LangGraph's state management system:

1. **Conversation State**
   ```python
   {
       "messages": List[Message],
       "max_research_loops": int,
       "initial_search_query_count": int
   }
   ```

2. **Research State**
   - Tracks research iterations
   - Maintains search results
   - Manages tool usage history

## Running the Backend

1. **Environment Setup**
   ```bash
   # Set up environment variables
   cp .env.example .env
   # Add your GEMINI_API_KEY
   ```

2. **Installation**
   ```bash
   # Install dependencies
   pip install -e .
   ```

3. **Development**
   ```bash
   # Run tests
   make test
   
   # Format code
   make format
   
   # Run linting
   make lint
   ```

4. **Deployment**
   - The backend can be deployed using the LangGraph API server
   - Configuration is managed through `langgraph.json`

## Agent Usage Example

```python
from agent import graph

# Initialize conversation
state = graph.invoke({
    "messages": [{"role": "user", "content": "Who won the euro 2024"}],
    "max_research_loops": 3,
    "initial_search_query_count": 3
})

# Continue conversation
state = graph.invoke({
    "messages": state["messages"] + [
        {"role": "user", "content": "How has the most titles?"}
    ]
})
```

## Adding New Tools

To add new tools to the agent:

1. Define the tool in `tools_and_schemas.py`
2. Add tool schema and implementation
3. Register the tool in the graph workflow
4. Update the agent's prompt to include tool usage

## Testing

The project includes a test suite and supports:

- Unit tests
- Integration tests
- Watch mode for development
- Test profiling

Run tests using the Makefile commands:
```bash
make test           # Run all tests
make test_watch    # Run tests in watch mode
make test_profile  # Run tests with profiling
```

## Development Guidelines

1. **Code Style**
   - Follow Google Python Style Guide
   - Use type hints
   - Document all modules and functions

2. **Testing**
   - Write unit tests for new features
   - Maintain test coverage
   - Use pytest for testing

3. **Configuration**
   - Use environment variables for secrets
   - Keep configuration in `configuration.py`
   - Use type-safe configuration

4. **Error Handling**
   - Implement proper error handling
   - Log errors appropriately
   - Provide meaningful error messages

## License

The project is licensed under the MIT License.
