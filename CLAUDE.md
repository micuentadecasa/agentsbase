# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fullstack agentic application generator built on LangGraph, React, and Google Gemini. The system creates intelligent agents that can coordinate complex business workflows. The project includes both a working example (backend/) and a generation system (backend_gen/) for creating new agent architectures.

## Key Architecture

### Dual Backend Structure
- `backend/` - Working LangGraph research agent (web search + reflection)
- `backend_gen/` - Generated agent workspace for new business cases
- `frontend/` - React/Vite interface for both backends

### Agent Generation System
The core innovation is an autonomous agent generation protocol that creates 10 different business case implementations:
1. Reads business requirements from `/docs/planning.md`
2. Generates complete LangGraph applications in `backend_gen/`
3. Follows structured phases with comprehensive testing
4. Accumulates knowledge in `/docs/tips.md`

## Development Commands

### Main Development
```bash
# Start both frontend and working backend
make dev

# Start frontend and generated backend
make gen

# Frontend only
make dev-frontend

# Working backend only  
make dev-backend

# Generated backend only
make dev-backend-gen
```

### Backend Development (Python/LangGraph)
```bash
cd backend  # or backend_gen
pip install -e .           # Install in editable mode
langgraph dev             # Start LangGraph dev server (port 2024)
python -m pytest tests/  # Run comprehensive tests
ruff check src/          # Lint code
```

### Frontend Development (React/TypeScript)
```bash
cd frontend
npm install              # Install dependencies
npm run dev             # Start Vite dev server (port 5173)
npm run build           # Build for production
npm run lint            # ESLint code
```

## Important File Patterns

### LangGraph Agent Structure
Required files in any backend implementation:
- `src/agent/state.py` - OverallState TypedDict definition
- `src/agent/graph.py` - Graph assembly with absolute imports
- `src/agent/nodes/` - Individual agent node functions
- `src/agent/tools_and_schemas.py` - Pydantic models and tools
- `src/agent/configuration.py` - LLM model configuration
- `langgraph.json` - Deployment configuration

### Critical Development Patterns

#### LLM Configuration Pattern
Always use configuration-based model selection:
```python
from agent.configuration import Configuration
from langchain_core.runnables import RunnableConfig

def node_function(state: OverallState, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.answer_model,  # Use configured model
        temperature=0,
        api_key=os.getenv("GEMINI_API_KEY")
    )
```

#### Import Requirements
- Use ABSOLUTE imports in graph.py: `from agent.nodes.x import y`
- Never use relative imports: `from .nodes.x import y` (breaks langgraph dev)
- Keep agent/__init__.py minimal to prevent circular imports

#### State Management
All agents share OverallState TypedDict with consistent field patterns:
```python
class OverallState(TypedDict):
    messages: List[Dict[str, Any]]
    # Domain-specific fields based on business case
    errors: Optional[List[str]]
```

## Testing Strategy

### Comprehensive Testing Phases
1. **Unit Testing** - Individual agent functions with real LLM calls
2. **Graph Compilation** - Import validation and graph building
3. **Server Testing** - LangGraph dev server with real execution
4. **API Integration** - REST endpoint validation
5. **Scenario Testing** - LangWatch Scenario framework for complex flows

### Critical Testing Commands
```bash
# Pre-server validation (prevents runtime failures)
python -c "from agent.graph import graph; print('Graph loads successfully')"
pip install -e .

# Server testing with proper cleanup
langgraph dev > langgraph.log 2>&1 &
SERVER_PID=$!
# ... testing logic ...
kill $SERVER_PID
```

## Environment Setup

### Required Environment Variables
```bash
# Backend (.env file)
GEMINI_API_KEY=your_gemini_api_key_here
LANGSMITH_API_KEY=your_langsmith_key_here  # Optional for tracing
```

### API Key Configuration
The system requires Google Gemini API key. LLM libraries do NOT automatically load from .env, so explicitly pass:
```python
api_key=os.getenv("GEMINI_API_KEY")
```

## Business Case Generation Protocol

The `/docs/planning.md` contains a comprehensive protocol for autonomous agent generation:

1. **Business Case Creation** - Generate diverse business scenarios
2. **Architecture Planning** - Design agent networks and workflows  
3. **Implementation** - Generate complete LangGraph applications
4. **Testing & Validation** - Comprehensive multi-phase testing
5. **Knowledge Accumulation** - Document patterns in `/docs/tips.md`

### Key Execution Principle
The system is designed for FULL AUTONOMY - no user confirmation required once started. Each business case follows structured phases with clear success criteria.

## Deployment

### Docker Build
```bash
docker build -t gemini-fullstack-langgraph -f Dockerfile .
```

### Production Server
```bash
GEMINI_API_KEY=<key> LANGSMITH_API_KEY=<key> docker-compose up
# Access at http://localhost:8123/app/
```

## Common Patterns

### Error Prevention
- Always test graph imports before server startup
- Use absolute imports in all graph files
- Validate environment variables before LLM calls
- Implement proper process cleanup in testing

### Agent Node Structure
Each agent node follows consistent patterns:
- Takes OverallState and RunnableConfig parameters
- Uses Configuration.from_runnable_config() for models
- Returns dict with state updates
- Handles errors gracefully with fallbacks

### Router vs Node Distinction
- Routers return string literals for graph navigation
- Nodes return dict state updates or None
- Only routers determine graph flow paths

This architecture enables systematic generation of complex multi-agent systems while maintaining consistency and reliability across different business domains.

## Using Gemini CLI for Large Codebase Analysis

When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive context window. Use `gemini -p` to leverage Google Gemini's large context capacity.

### File and Directory Inclusion Syntax

Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the gemini command:

#### Examples:

**Single file analysis:**
```bash
gemini -p "@src/main.py Explain this file's purpose and structure"
```

**Multiple files:**
```bash
gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"
```

**Entire directory:**
```bash
gemini -p "@src/ Summarize the architecture of this codebase"
```

**Multiple directories:**
```bash
gemini -p "@src/ @tests/ Analyze test coverage for the source code"
```

**Current directory and subdirectories:**
```bash
gemini -p "@./ Give me an overview of this entire project"

# Or use --all_files flag:
gemini --all_files -p "Analyze the project structure and dependencies"
```

### Implementation Verification Examples

**Check if a feature is implemented:**
```bash
gemini -p "@src/ @lib/ Has dark mode been implemented in this codebase? Show me the relevant files and functions"
```

**Verify authentication implementation:**
```bash
gemini -p "@src/ @middleware/ Is JWT authentication implemented? List all auth-related endpoints and middleware"
```

**Check for specific patterns:**
```bash
gemini -p "@src/ Are there any React hooks that handle WebSocket connections? List them with file paths"
```

**Verify error handling:**
```bash
gemini -p "@src/ @api/ Is proper error handling implemented for all API endpoints? Show examples of try-catch blocks"
```

**Check for rate limiting:**
```bash
gemini -p "@backend/ @middleware/ Is rate limiting implemented for the API? Show the implementation details"
```

**Verify caching strategy:**
```bash
gemini -p "@src/ @lib/ @services/ Is Redis caching implemented? List all cache-related functions and their usage"
```

**Check for specific security measures:**
```bash
gemini -p "@src/ @api/ Are SQL injection protections implemented? Show how user inputs are sanitized"
```

**Verify test coverage for features:**
```bash
gemini -p "@src/payment/ @tests/ Is the payment processing module fully tested? List all test cases"
```

### When to Use Gemini CLI

Use `gemini -p` when:
- Analyzing entire codebases or large directories
- Comparing multiple large files
- Need to understand project-wide patterns or architecture
- Current context window is insufficient for the task
- Working with files totaling more than 100KB
- Verifying if specific features, patterns, or security measures are implemented
- Checking for the presence of certain coding patterns across the entire codebase

### Important Notes

- Paths in `@` syntax are relative to your current working directory when invoking gemini
- The CLI will include file contents directly in the context
- No need for --yolo flag for read-only analysis
- Gemini's context window can handle entire codebases that would overflow Claude's context
- When checking implementations, be specific about what you're looking for to get accurate results

## Autonomous Development Instructions

The following sections contain the comprehensive instructions for autonomous development of LangGraph applications based on planning_old.md and tips.md.

### ENHANCED LANGGRAPH PROJECT CONFIGURATION & CLI INTEGRATION

#### 0. ITERATIVE BUSINESS CASE EXECUTION PROTOCOL

##### Master Execution Loop
**Execute 10 complete business case iterations** to build comprehensive knowledge base **AUTONOMOUSLY, WITHOUT USER CONFIRMATION AT ANY STEP**:

1. **Business Case Generation Phase**
   - Think creatively about a new agentic business case
   - Ensure each case explores different patterns (see variety matrix below)
   - Document the business case rationale and expected challenges
   - Create `/tasks/iteration_X_business_case.md` with detailed specification

2. **Implementation Phase**
   - Follow standard execution phases (0-3) for the business case
   - Apply lessons learned from `/docs/tips.md` proactively
   - Document new patterns and solutions discovered

3. **Error Learning Phase**
   - Every time an error is encountered and fixed, update `/docs/tips.md`
   - Follow the Enhanced Tips Format
   - Review existing tips before writing new code or tests

4. **Knowledge Accumulation Phase**
   - After each iteration, update `/docs/patterns_learned.md`
   - Document successful architectural decisions
   - Note business case complexity vs implementation patterns

For each round and for each phase write a file in /tasks indicating the steps that have to be done and the instructions for each specific phase, and then follow what that file says. Include all the examples of code or tips for each phase in the file.

**NEVER use mock APIs, never, period.**

##### Business Case Variety Matrix
Ensure coverage across these dimensions over 10 iterations:

| Dimension | Options | Target Coverage |
|-----------|---------|----------------|
| **Domain** | Healthcare, Finance, Education, E-commerce, Legal, Manufacturing, Research, Content, Operations | ≥7 domains |
| **Agent Count** | 2, 3, 4-6, 7+ | All ranges |
| **Architecture** | Monolithic, Supervisor, Hierarchical, Network, Custom | All types |
| **Data Sources** | APIs, Files, Databases, Web scraping, User input | ≥4 sources |
| **Output Types** | Text, Files, API calls, Database updates, Notifications | ≥4 types |
| **Complexity** | Simple linear, Conditional branching, Loops, Error recovery, Human-in-loop | All levels |

#### 1. EXECUTION PHASES & SUCCESS CRITERIA

##### Phase -1: Business Case Generation
**Objective**: Generate creative, diverse business case for current iteration
**Success Criteria**:
- Business case documented in `/tasks/iteration_X_business_case.md`
- Case fits variety matrix requirements
- Clear agent roles and responsibilities defined
- Expected technical challenges identified
- Success metrics established

##### Phase 0: Workspace Initialization
**Objective**: Clean slate preparation with tips consultation. **This phase and all subsequent phases proceed automatically, without user confirmation.**
```bash
# 1. Hard reset tasks directory
rm -rf /tasks
mkdir -p /tasks/artifacts

# 2. Hard reset backend_gen directory  
rm -rf /backend_gen

# 3. Copy backend to backend_gen
cp -r /backend /backend_gen

# 4. Verify structure
ls -la /backend_gen/src/agent/

# 5. Install dependencies
cd /backend_gen && pip install -e .
```

##### Phase 1: Architecture Planning & Specification
**Objective**: Complete project specification before any implementation, incorporating lessons learned.

**Critical Tips for Node Implementation**:
- **TIP #012**: Use Configuration.from_runnable_config() instead of hardcoded models
- **TIP #008**: LangGraph Agent Function Signature must use RunnableConfig
- **TIP #010**: State management with None values using safe patterns
- **TIP #006**: Use absolute imports in graph.py to avoid server startup failures

**MANDATORY LLM Call Pattern (Using Configuration - TIP #012)**:
```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
import os
from agent.state import OverallState
from agent.configuration import Configuration

def node_function(state: OverallState, config: RunnableConfig) -> dict:
    # ✅ CRITICAL: Get configuration from RunnableConfig (TIP #012)
    configurable = Configuration.from_runnable_config(config)
    
    # ✅ CRITICAL: Use configured model, not hardcoded
    llm = ChatGoogleGenerativeAI(
        model=configurable.answer_model,  # Use configured model!
        temperature=0,  # For deterministic responses
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    # Example: Format a prompt using state
    prompt = f"Schedule appointment for {state['patient_info']['name']} with available doctors."
    result = llm.invoke(prompt)
    state["messages"].append({"role": "agent", "content": result.content})
    return state
```

**CRITICAL: Use ABSOLUTE imports only (relative imports will fail in langgraph dev)**:
```python
# ❌ WRONG - Relative imports (will cause server startup failure)
# from .nodes.clinical_data_collector import clinical_data_collector_agent

# ✅ CORRECT - Absolute imports (required for langgraph dev server)
from agent.nodes.clinical_data_collector import clinical_data_collector_agent
```

##### Phase 2: Implementation & Code Generation
**Objective**: Generate all required code components using accumulated knowledge.

**Mandatory Files Checklist**:
- [ ] `state.py` - OverallState TypedDict
- [ ] `tools_and_schemas.py` - Pydantic models/tools
- [ ] `nodes/` directory with individual node files
- [ ] `graph.py` - Complete graph assembly
- [ ] `langgraph.json` - Deployment configuration
- [ ] `tests/` directory with comprehensive unit tests

##### Phase 3: Testing & Validation
**Objective**: Comprehensive testing and error resolution with knowledge capture.

###### Phase 3.1: Unit Testing & Component Validation
**Success Criteria**:
- All unit tests pass with real LLM calls, file operations, and computations
- Individual agent functions work correctly with proper signatures
- Pydantic models handle data validation and conversion properly
- **LLM conversations logged** for debugging and verification
- Error handling mechanisms tested with fallback data

###### Phase 3.2: Graph Compilation & Import Validation  
**Success Criteria**:
- Graph compiles and imports successfully without errors
- All node imports execute without circular dependencies
- State schema is valid TypedDict structure
- **Pre-server validation** prevents import errors at runtime
- Package installation completes successfully

###### Phase 3.3: LangGraph Server Testing & Real Execution
**CRITICAL LESSONS LEARNED**: Import errors only surface when server actually runs, not during unit tests.

**MANDATORY PRE-SERVER CHECKS**:
```bash
# 1. CRITICAL: Fix relative imports in graph.py BEFORE server testing
# 2. CRITICAL: Fix fake LLM imports in utils.py
# 3. CRITICAL: Test graph loading BEFORE server
cd /backend_gen
python -c "from agent.graph import graph; print('Graph loads:', type(graph))"
# 4. CRITICAL: Install package in editable mode
pip install -e .
```

**Success Criteria**:
- `langgraph dev` server starts without errors on port 2024
- Server logs show no ImportError, ModuleNotFoundError, or relative import issues
- **Real LLM execution** processes requests with actual API calls (not mocks)
- Graph execution transitions through all states successfully

### LangGraph Development Tips & Error Solutions

#### Tips Consultation Protocol
**MANDATORY**: Before writing any code or tests, consult this file for:
1. **Pre-Code Review**: Check tips related to the component being implemented
2. **Error Pattern Matching**: When an error occurs, first check if it's documented
3. **Solution Application**: Apply documented solutions before creating new ones
4. **Pattern Recognition**: Identify if the current business case matches previous patterns

#### Critical Development Tips

##### TIP #001: Project Initialization Template
**Category**: Architecture | **Severity**: Critical

**Solution Implementation**:
```bash
# Always start with clean reset
rm -rf tasks
mkdir -p tasks/artifacts
rm -rf backend_gen
cp -r backend backend_gen
cd backend_gen && pip install -e .
```

##### TIP #006: Critical LangGraph Server Import Errors and Solutions
**Category**: Development | **Severity**: Critical

**Problem Description**: Multiple specific import errors that only surface when `langgraph dev` server actually runs:
1. **Relative Import Error**: `ImportError: attempted relative import with no known parent package`
2. **Fake Model Import Errors**: Cannot import FakeListChatModel, FakeChatModel
3. **Module Path Errors**: `No module named 'langchain_core.language_models.llm'`

**Solution Implementation**:

Fix Relative Imports in graph.py:
```python
# ❌ WRONG - Relative imports (will fail in server)
from .nodes.clinical_data_collector import clinical_data_collector_agent

# ✅ CORRECT - Absolute imports (works in server)
from agent.nodes.clinical_data_collector import clinical_data_collector_agent
```

Fix Fake LLM Imports:
```python
def get_llm():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Try real LLM first
    if os.getenv("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            api_key=os.getenv("GEMINI_API_KEY")
        )
    
    # Fallback to working fake model
    try:
        from langchain_core.language_models.fake import FakeListLLM
        return FakeListLLM(responses=["Mock response"])
    except ImportError:
        class SimpleFakeLLM:
            def invoke(self, prompt):
                class Response:
                    content = "Mock LLM response for testing"
                return Response()
        return SimpleFakeLLM()
```

##### TIP #012: Proper Configuration Usage in LangGraph Nodes
**Category**: Development | **Severity**: High

**Problem Description**: Nodes that hardcode LLM model names instead of using the configuration system create inflexible applications.

**Solution Implementation**:
```python
# ❌ WRONG - Hardcoded model
def my_agent_node(state: OverallState, config: Dict[str, Any]) -> Dict[str, Any]:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # HARDCODED - BAD!
        temperature=0.1,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

# ✅ CORRECT - Use configuration
from agent.configuration import Configuration

def my_agent_node(state: OverallState, config: RunnableConfig) -> Dict[str, Any]:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.answer_model,  # Use configured model
        temperature=0.1,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
```

**Configuration Fields Available**:
- `query_generator_model`: Default "gemini-2.0-flash"
- `reflection_model`: Default "gemini-2.5-flash-preview-04-17" 
- `answer_model`: Default "gemini-1.5-flash-latest"
- `number_of_initial_queries`: Default 3
- `max_research_loops`: Default 2

##### TIP #005: The Critical Gap Between Unit Tests and Real LLM Endpoint Testing
**Category**: Testing | **Severity**: Critical

**Problem Description**: Mock tests can pass while real API endpoints fail catastrophically. Unit tests use mocked dependencies, so they pass even when import statements are incorrect for the actual runtime environment.

**Solution Implementation**:
```bash
# STEP 1: Test graph loading in server context BEFORE unit tests
cd /backend_gen
python -c "from agent.graph import graph; print('✅ Graph loads successfully:', type(graph))"

# STEP 2: Start server and check logs for import errors
nohup langgraph dev > langgraph.log 2>&1 &
sleep 10

# CRITICAL: Check for import errors in server logs
if grep -q "ImportError\|ModuleNotFoundError" langgraph.log; then
    echo "❌ CRITICAL: Server has import errors"
    grep -A3 -B3 "Error" langgraph.log
    exit 1
fi
```

**Prevention Strategy**: Always test in this order:
1. **Graph import test** in server context first
2. **Server startup** with log monitoring for import errors  
3. **Real endpoint calls** with actual data payloads
4. **LLM response validation** in final state
5. **Only then** run unit tests as confirmation

##### Additional Key Tips

**TIP #008**: LangGraph Agent Function Signature Inconsistencies - Some agents require `config` parameter while others don't.

**TIP #009**: Pydantic Model Mutability - Use `.model_dump()` for dictionary conversion in Pydantic v2.

**TIP #010**: State management with None values - Always check for None before list operations: `existing_errors = state.get("errors") or []`

**TIP #011**: LangGraph Server Port Conflicts - Use port cleanup and dynamic port selection for testing.

#### Current Development Status

**Iteration 6: Manufacturing Supply Chain Optimization** ✅ **PHASE 3 COMPLETED**
- **Business Case**: Manufacturing Supply Chain Optimization ($24.8B Market)
- **Architecture**: Network (5 Agents) - Supply Chain Coordinator, Supplier Intelligence, Production Planning, Logistics Optimization, Quality Assurance
- **Status**: Ready for Phase 4 (Production Integration)

**Next Iteration Ready**: **Iteration 7** - Ready to commence with autonomous protocol