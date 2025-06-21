# ENHANCED LANGGRAPH PROJECT CONFIGURATION & CLI INTEGRATION

---

> **Purpose of this document** – Provide a comprehensive, structured guide for autonomous AI agents to develop LangGraph applications with clear execution phases, validation checkpoints, and error recovery patterns.

---

## 1. ROLE DEFINITION & OPERATIONAL PARAMETERS

### Primary Role
**You are an expert-level, autonomous AI Project Manager and Lead Developer** with the following operational parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Autonomy Level** | Full | No user confirmation required after initial start |
| **State Tracking** | File-system only | All progress tracked through files |
| **Error Handling** | Self-correcting | Must fix errors and document solutions |
| **Completion Standard** | Production-ready | All code must pass tests and run without errors |

### Mission Statement
Orchestrate and execute the development of a LangGraph application based on provided documentation **fully autonomously**, ensuring:
- Complete blueprint compliance
- Robust error handling and recovery
- Comprehensive testing and validation
- Production-ready deployment artifacts

### Available Tools
- `read_file` - Read existing files
- `write_file` - Create/modify files
- `execute_shell_command` - Run terminal commands
- File system operations for project management

---

## 2. EXECUTION PHASES & SUCCESS CRITERIA

### Phase 0: Workspace Initialization
**Objective**: Clean slate preparation
**Success Criteria**: 
- `/tasks` directory completely reset
- `/backend_gen` directory completely reset
- `/backend/` successfully copied to `/backend_gen/`
- Environment validated and ready

**Validation Commands**:
```bash
ls -la /tasks  # Should be empty or non-existent
ls -la /backend_gen  # Should contain copied backend structure
```

### Phase 1: Architecture Planning & Specification
**Objective**: Complete project specification before any implementation
**Success Criteria**:
- All documentation internalized and understood
- `/tasks/01_define-graph-spec.md` created with detailed execution plan
- `/tasks/artifacts/graph_spec.yaml` generated with complete architecture
- Business case framing completed
- Testing strategy defined

**Critical Rule**: NO implementation code until this phase is 100% complete

### Phase 2: Implementation & Code Generation
**Objective**: Generate all required code components
**Success Criteria**:
- All mandatory files created under `/backend_gen/src/agent/`
- LLM integration properly configured
- All nodes follow MANDATORY LLM Call Pattern
- Graph assembly completed
- Import validation successful

**Mandatory Files Checklist**:
- [ ] `state.py` - OverallState TypedDict
- [ ] `tools_and_schemas.py` - Pydantic models/tools
- [ ] `nodes/` directory with individual node files
- [ ] `graph.py` - Complete graph assembly
- [ ] `langgraph.json` - Deployment configuration
- [ ] `tests/` directory with comprehensive unit tests
- [ ] `tests/test_agents.py` - Individual agent unit tests
- [ ] `tests/test_tools.py` - Tool validation tests
- [ ] `tests/test_schemas.py` - Pydantic model tests

### Phase 3: Testing & Validation
**Objective**: Comprehensive testing and error resolution
**Success Criteria**:
- Graph compiles without errors
- Direct Python testing passes
- API testing via langgraph dev succeeds
- All test scenarios validated
- Server logs show no errors

---

## 3. CORE PRINCIPLES & NON-NEGOTIABLES

### Architectural Principles
1. **Planning First** - No implementation until complete planning phase
2. **Blueprint Compliance** - Every artifact must conform to `/docs/blueprint_backend.md`
3. **Full Autonomy** - Proceed without user interaction once plan exists
4. **Error Documentation** - Every error must be logged with solution in `/docs/tips.md`
5. **Router Rule** - Only router returns sentinel strings; nodes return dict, NOTHING, or raise

### Technical Standards
1. **Environment Handling**:
   - Source: `backend/.env`
   - Target: `backend_gen/.env`
   - Validation required before graph testing
   - Single API key request if missing

2. **LLM Configuration**:
   - Use providers from `backend/src/agent/configuration.py`.
   - **Note**: For any node making LLM calls, ensure the API key from the `.env` file is explicitly passed to the constructor (e.g., `api_key=os.getenv("GEMINI_API_KEY")`). The library will not load it automatically.
   - Set `temperature=0` for deterministic nodes
   - Implement proper error handling and retries

3. **Command Standards**:
   - Always use `langgraph dev`, never `langgraph up`
   - Use context7 for latest LangGraph documentation
   - Validate with `pip install -e .` before testing

---

## 4. STREAMLINED MASTER WORKFLOW

### Pre-Execution Checklist
Before starting any phase, verify:
- [ ] All required documentation is accessible
- [ ] Environment variables are properly configured
- [ ] Previous phase completion criteria are met
- [ ] Error ledger (`/docs/tips.md`) has been reviewed

### Phase 0: Workspace Initialization
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
```
remember to run pip install -e . in the backend_gen directory.

### Phase 1: Node Specification & Flow Design

#### 1.1 Documentation Internalization
- Read and understand all provided documentation
- Identify key requirements and constraints
- Map business requirements to technical architecture

#### 1.2 Task Definition
Create `/tasks/01_define-graph-spec.md` with:
- Detailed task description
- Expected outputs
- Validation criteria
- Dependencies

#### 1.3 Architecture Specification
Generate `/tasks/artifacts/graph_spec.yaml` following the Business-Case Checklist:

**Required Sections**:
1. **Business Case Framing**
   - High-level goal definition
   - Core competencies identification
   - Architecture choice (centralized vs distributed)
   - External API requirements
   - Data flow mapping
   - Testing strategy

2. **Architecture Selection**
   Use the decision table to choose:
   - Monolithic graph (single linear task, few tools)
   - Supervisor (2-6 specialized agents, centralized decisions)
   - Hierarchical (>6 agents, multiple domains)
   - Network (free agent communication)
   - Custom workflow (deterministic pipeline)

3. **Agent & Tool Specification**
   - Agent roles and responsibilities
   - Concrete tool assignments
   - Tool-calling vs graph-node differentiation

4. **State & Message Design**
   - Shared vs private channels
   - InjectedState requirements
   - Data flow patterns

5. **Testing Plan**
   - Unit test scenarios
   - Integration test patterns
   - API test specifications

### Phase 2: Direct Code Implementation

#### 2.1 State Definition
**File**: `/backend_gen/src/agent/state.py`
```python
from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional

class OverallState(TypedDict):
    # Define based on graph_spec.yaml requirements
    messages: List[Dict[str, Any]]
    # Add other state fields as needed
```

#### 2.2 Tools and Schemas
**File**: `/backend_gen/src/agent/tools_and_schemas.py`
- Pydantic models for data validation
- Tool wrapper functions
- Schema definitions for LLM interactions

#### 2.3 Node Implementation
**Directory**: `/backend_gen/src/agent/nodes/`

**MANDATORY LLM Call Pattern**:
```python
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def node_function(state: OverallState) -> Dict[str, Any]:
    # Initialize LLM with proper configuration
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # or from config
        temperature=0,  # For deterministic responses
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    # Implement node logic
    # Return proper dict or raise exception
    return {"updated_field": "value"}
```

#### 2.4 Graph Assembly
**File**: `/backend_gen/src/agent/graph.py`
```python
from langgraph.graph import StateGraph, START, END
from .state import OverallState
from .nodes import node1, node2, router

def build_graph():
    builder = StateGraph(OverallState)
    
    # Add nodes (not router)
    builder.add_node("node1", node1)
    builder.add_node("node2", node2)
    
    # Add edges with router logic
    builder.add_conditional_edges(
        START,
        router,
        {"option1": "node1", "option2": "node2"}
    )
    
    return builder.compile()

# Important: Instantiate the graph
graph = build_graph()
```

#### 2.6 Unit Test Implementation
**Directory**: `/backend_gen/tests/`

Create comprehensive unit tests for each component:

**File**: `/backend_gen/tests/test_agents.py`
```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from agent.nodes.prompt_enhancer import prompt_enhancer_node
from agent.nodes.answer_generator import answer_generator_node
from agent.state import OverallState

class TestPromptEnhancerAgent:
    """Unit tests for the prompt enhancer agent"""
    
    def setup_method(self):
        """Setup test data for each test"""
        self.base_state = {
            "messages": [{"role": "human", "content": "What is AI?"}],
            "original_question": "What is AI?",
            "enhanced_prompt": None,
            "final_answer": None
        }
    
    @patch('agent.nodes.prompt_enhancer.ChatGoogleGenerativeAI')
    def test_prompt_enhancer_basic_functionality(self, mock_llm_class):
        """Test basic prompt enhancement functionality"""
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Enhanced question: What is artificial intelligence and how does it work?"
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        # Execute agent
        result = prompt_enhancer_node(self.base_state)
        
        # Validate results
        assert result is not None
        assert "enhanced_prompt" in result
        assert result["enhanced_prompt"] is not None
        assert len(result["enhanced_prompt"]) > len(self.base_state["original_question"])
        
        # Verify LLM was called correctly
        mock_llm_class.assert_called_once()
        mock_llm.invoke.assert_called_once()
    
    @patch('agent.nodes.prompt_enhancer.ChatGoogleGenerativeAI')
    def test_prompt_enhancer_error_handling(self, mock_llm_class):
        """Test error handling when LLM fails"""
        # Mock LLM to raise exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API Error")
        mock_llm_class.return_value = mock_llm
        
        # Execute agent and expect proper error handling
        with pytest.raises(Exception) or pytest.warns(UserWarning):
            result = prompt_enhancer_node(self.base_state)
            # If no exception, should have fallback behavior
            if result:
                assert "enhanced_prompt" in result
    
    def test_prompt_enhancer_input_validation(self):
        """Test input validation and edge cases"""
        # Test with empty question
        empty_state = self.base_state.copy()
        empty_state["original_question"] = ""
        
        # Should handle gracefully
        with patch('agent.nodes.prompt_enhancer.ChatGoogleGenerativeAI'):
            result = prompt_enhancer_node(empty_state)
            assert result is not None
        
        # Test with very long question
        long_state = self.base_state.copy()
        long_state["original_question"] = "What is AI? " * 1000
        
        with patch('agent.nodes.prompt_enhancer.ChatGoogleGenerativeAI'):
            result = prompt_enhancer_node(long_state)
            assert result is not None

class TestAnswerGeneratorAgent:
    """Unit tests for the answer generator agent"""
    
    def setup_method(self):
        """Setup test data for each test"""
        self.base_state = {
            "messages": [{"role": "human", "content": "What is AI?"}],
            "original_question": "What is AI?",
            "enhanced_prompt": "What is artificial intelligence and how does it work?",
            "final_answer": None
        }
    
    @patch('agent.nodes.answer_generator.ChatGoogleGenerativeAI')
    def test_answer_generator_basic_functionality(self, mock_llm_class):
        """Test basic answer generation functionality"""
        # Mock LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Artificial Intelligence (AI) is a comprehensive field..."
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        # Execute agent
        result = answer_generator_node(self.base_state)
        
        # Validate results
        assert result is not None
        assert "final_answer" in result
        assert result["final_answer"] is not None
        assert len(result["final_answer"]) > 0
        
        # Verify LLM was called with enhanced prompt
        mock_llm_class.assert_called_once()
        mock_llm.invoke.assert_called_once()
    
    @patch('agent.nodes.answer_generator.ChatGoogleGenerativeAI')
    def test_answer_generator_quality_validation(self, mock_llm_class):
        """Test answer quality validation"""
        # Mock high-quality response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "A comprehensive answer with detailed explanation..."
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        result = answer_generator_node(self.base_state)
        
        # Validate answer quality metrics
        assert len(result["final_answer"]) > 20  # Minimum length
        assert "." in result["final_answer"]  # Contains sentences
        
    def test_answer_generator_missing_enhanced_prompt(self):
        """Test behavior when enhanced prompt is missing"""
        state_no_prompt = self.base_state.copy()
        state_no_prompt["enhanced_prompt"] = None
        
        with patch('agent.nodes.answer_generator.ChatGoogleGenerativeAI'):
            # Should handle gracefully, possibly use original question
            result = answer_generator_node(state_no_prompt)
            assert result is not None
            assert "final_answer" in result
```

**File**: `/backend_gen/tests/test_tools.py`
```python
import pytest
from unittest.mock import Mock, patch
from agent.tools_and_schemas import (
    EnhancePromptTool, 
    GenerateAnswerTool,
    QuestionSchema,
    AnswerSchema
)

class TestToolFunctionality:
    """Test individual tool operations"""
    
    def test_enhance_prompt_tool_structure(self):
        """Test prompt enhancement tool structure"""
        tool = EnhancePromptTool()
        
        # Validate tool attributes
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert callable(tool.func) if hasattr(tool, 'func') else True
    
    @patch('agent.tools_and_schemas.ChatGoogleGenerativeAI')
    def test_enhance_prompt_tool_execution(self, mock_llm_class):
        """Test prompt enhancement tool execution"""
        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Enhanced: What is AI and its applications?"
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        # Execute tool
        tool = EnhancePromptTool()
        if hasattr(tool, 'func'):
            result = tool.func("What is AI?")
            assert result is not None
            assert len(result) > 0
    
    def test_generate_answer_tool_structure(self):
        """Test answer generation tool structure"""
        tool = GenerateAnswerTool()
        
        # Validate tool attributes
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert callable(tool.func) if hasattr(tool, 'func') else True
    
    @patch('agent.tools_and_schemas.ChatGoogleGenerativeAI')
    def test_generate_answer_tool_execution(self, mock_llm_class):
        """Test answer generation tool execution"""
        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "AI is a field of computer science..."
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        # Execute tool
        tool = GenerateAnswerTool()
        if hasattr(tool, 'func'):
            result = tool.func("What is artificial intelligence?")
            assert result is not None
            assert len(result) > 0

class TestToolIntegration:
    """Test tool integration with agents"""
    
    def test_tools_available_to_agents(self):
        """Verify agents can access their required tools"""
        from agent.nodes.prompt_enhancer import prompt_enhancer_node
        from agent.nodes.answer_generator import answer_generator_node
        
        # This test ensures tools are properly imported and accessible
        # The actual functionality is tested in agent tests
        assert prompt_enhancer_node is not None
        assert answer_generator_node is not None
    
    def test_tool_error_propagation(self):
        """Test how tool errors are handled by agents"""
        # Test that tool failures are properly caught and handled
        # This integrates with the agent error handling tests
        pass
```

**File**: `/backend_gen/tests/test_schemas.py`
```python
import pytest
from pydantic import ValidationError
from agent.tools_and_schemas import (
    QuestionSchema,
    AnswerSchema,
    EnhancedPromptSchema
)

class TestPydanticSchemas:
    """Test Pydantic model validation"""
    
    def test_question_schema_valid_input(self):
        """Test question schema with valid inputs"""
        valid_questions = [
            "What is AI?",
            "How does machine learning work?",
            "Explain neural networks in simple terms."
        ]
        
        for question in valid_questions:
            schema = QuestionSchema(question=question)
            assert schema.question == question
            assert len(schema.question) > 0
    
    def test_question_schema_invalid_input(self):
        """Test question schema with invalid inputs"""
        invalid_inputs = [
            "",  # Empty string
            None,  # None value
            "   ",  # Only whitespace
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ValidationError):
                QuestionSchema(question=invalid_input)
    
    def test_answer_schema_valid_input(self):
        """Test answer schema with valid inputs"""
        valid_answer = "AI is a field of computer science that focuses on creating intelligent machines."
        
        schema = AnswerSchema(answer=valid_answer)
        assert schema.answer == valid_answer
        assert len(schema.answer) > 0
    
    def test_answer_schema_validation_rules(self):
        """Test answer schema validation rules"""
        # Test minimum length requirement
        short_answer = "AI."
        with pytest.raises(ValidationError):
            AnswerSchema(answer=short_answer)
        
        # Test valid longer answer
        good_answer = "AI is a comprehensive field that encompasses machine learning, natural language processing, and robotics."
        schema = AnswerSchema(answer=good_answer)
        assert schema.answer == good_answer
    
    def test_enhanced_prompt_schema(self):
        """Test enhanced prompt schema"""
        original = "What is AI?"
        enhanced = "What is artificial intelligence, including its main branches, applications, and current limitations?"
        
        schema = EnhancedPromptSchema(
            original_question=original,
            enhanced_prompt=enhanced
        )
        
        assert schema.original_question == original
        assert schema.enhanced_prompt == enhanced
        assert len(schema.enhanced_prompt) > len(schema.original_question)

class TestSchemaIntegration:
    """Test schema integration with the overall system"""
    
    def test_state_schema_compatibility(self):
        """Test that schemas work with the state management"""
        from agent.state import OverallState
        
        # Test that our schemas are compatible with the state structure
        state_data = {
            "messages": [{"role": "human", "content": "What is AI?"}],
            "original_question": "What is AI?",
            "enhanced_prompt": "What is artificial intelligence and its applications?",
            "final_answer": "AI is a field of computer science..."
        }
        
        # This should not raise validation errors
        # The actual validation depends on how OverallState is implemented
        assert isinstance(state_data, dict)
        assert all(key in state_data for key in ["messages", "original_question"])
```

**File**: `/backend_gen/tests/conftest.py`
```python
import pytest
import os
from unittest.mock import Mock
from dotenv import load_dotenv

# Load test environment variables
load_dotenv(dotenv_path=".env")

@pytest.fixture
def mock_llm():
    """Provide a mock LLM for testing"""
    mock = Mock()
    mock.invoke.return_value = Mock(content="Mocked LLM response")
    return mock

@pytest.fixture
def sample_state():
    """Provide sample state for testing"""
    return {
        "messages": [{"role": "human", "content": "What is AI?"}],
        "original_question": "What is AI?",
        "enhanced_prompt": None,
        "final_answer": None
    }

@pytest.fixture
def enhanced_state():
    """Provide state with enhanced prompt for testing"""
    return {
        "messages": [{"role": "human", "content": "What is AI?"}],
        "original_question": "What is AI?",
        "enhanced_prompt": "What is artificial intelligence and how does it work?",
        "final_answer": None
    }

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    os.environ.setdefault("GEMINI_API_KEY", "test-key-for-testing")
    yield
    # Cleanup if needed
```
**File**: `/backend_gen/langgraph.json`
```json
{
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "dependencies": []
}
```

### Phase 3: Testing & Validation

#### 3.0 Unit Testing Execution
```bash
# Run all unit tests first
cd /backend_gen
python -m pytest tests/test_agents.py -v
python -m pytest tests/test_tools.py -v  
python -m pytest tests/test_schemas.py -v

# Run with coverage report
python -m pytest tests/ --cov=agent --cov-report=html --cov-report=term
```

**Unit Test Success Criteria**:
- [ ] All agent tests pass with mocked dependencies
- [ ] All tool tests validate functionality and error handling
- [ ] All schema tests cover validation rules and edge cases
- [ ] Test coverage > 80% for all agent and tool code
- [ ] No real API calls during unit testing (all mocked)

#### 3.1 Validation Tasks
```bash
# Install and verify
cd /backend_gen
pip install -e .

# Test imports
python -c "from agent.graph import build_graph; build_graph()"

# Verify graph structure
python -c "
from agent.graph import graph
print('Graph name:', graph.name if hasattr(graph, 'name') else 'agent')
print('Nodes:', list(graph.nodes.keys()) if hasattr(graph, 'nodes') else 'Check compilation')
"
```

#### 3.2 Direct Testing Script
**Note**: This test makes live API calls. Ensure your testing script loads environment variables (e.g., `from dotenv import load_dotenv; load_dotenv()`) so the API key from `.env` is available.

Create `/backend_gen/test_direct.py`:
```python
import pytest
from agent.graph import graph
from agent.state import OverallState

def test_graph_direct():
    """Test graph execution directly without API"""
    initial_state = {
        "messages": [{"role": "human", "content": "Test message"}]
    }
    
    # Execute graph
    result = graph.invoke(initial_state)
    
    # Validate results
    assert result is not None
    assert "messages" in result
    print("Direct test passed:", result)

if __name__ == "__main__":
    test_graph_direct()
```

#### 3.3 API Server Validation
**Critical Note**: Do not use a `pytest` script to validate the running server. The `langgraph dev` server should be validated directly with a client tool like `curl` to simulate real API interaction.

**Validation Steps**:
1.  Start the server in one terminal:
    ```bash
    cd /backend_gen
    langgraph dev --port 8000
    ```

2.  In a second terminal, send a request using `curl`. The `-N` or `--no-buffer` flag is **mandatory** to handle the streaming response correctly and keep the connection open.
    ```bash
    curl -X POST -N -H "Content-Type: application/json" \
    -d '{"assistant_id": "agent", "input": {"messages": [{"role": "human", "content": "What is AI?"}]}, "stream_mode": "updates"}' \
    http://127.0.0.1:8000/runs
    ```

**Expected Output**:
- The `curl` command will stay connected and print the server's real-time log output as the graph executes.
- You will see the print statements from each node (`---PROMPT ENHANCER NODE---`, etc.).
- This confirms the server is receiving requests and executing the graph end-to-end. The stream data itself is logged to `stderr` by the server, not sent as clean `data:` events over `stdout` in this development mode.


### Phase 3: Testing & Validation

#### 3.0 Unit Testing Execution
// ... existing code ...

---

## 5. COMPREHENSIVE TESTING STRATEGY

### Unit Testing Requirements

#### Agent Testing Standards
Each agent must have unit tests covering:

1. **Functionality Testing**
   - Core logic validation with mocked dependencies
   - Input/output transformation verification
   - State modification correctness

2. **Error Handling Testing**
   - LLM API failures and timeouts
   - Invalid input handling
   - Network connectivity issues
   - Graceful degradation scenarios

3. **Tool Integration Testing**
   - Tool accessibility and invocation
   - Tool response processing
   - Tool error propagation

4. **Edge Case Testing**
   - Empty or null inputs
   - Extremely long inputs
   - Special characters and encoding
   - Boundary conditions

#### Tool Testing Standards
Each tool must have unit tests covering:

1. **Interface Testing**
   - Proper tool structure and attributes
   - Function signature validation
   - Parameter handling

2. **Execution Testing**
   - Core functionality with various inputs
   - Response format validation
   - Performance within acceptable limits

3. **Error Scenarios**
   - Invalid parameters
   - External service failures
   - Timeout handling
   - Exception propagation

#### Schema Testing Standards
All Pydantic schemas must have tests covering:

1. **Validation Rules**
   - Valid input acceptance
   - Invalid input rejection
   - Type checking and conversion

2. **Business Logic**
   - Field relationships and dependencies
   - Custom validators
   - Data transformation

3. **Integration Compatibility**
   - State management compatibility
   - API serialization/deserialization
   - Cross-schema relationships

### Unit Test Implementation Guidelines

#### Mocking Strategy
- **LLM Services**: Always mock external LLM API calls
- **File Operations**: Mock file system interactions
- **Network Calls**: Mock all external network requests
- **Environment Variables**: Use test fixtures for configuration

#### Test Organization
```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_agents.py           # Agent-specific unit tests
├── test_tools.py            # Tool functionality tests
├── test_schemas.py          # Pydantic model validation tests
├── test_integration.py      # Component integration tests
└── test_graph.py           # Graph compilation and structure tests
```

#### Coverage Requirements
- **Minimum Coverage**: 80% for all agent and tool code
- **Critical Path Coverage**: 100% for core business logic
- **Error Path Coverage**: All exception scenarios tested
- **Branch Coverage**: All conditional logic paths covered

### Testing Task Requirements

Create the following additional test tasks:

1. **Unit Test Generation Task**
   - Generate comprehensive unit tests for each agent
   - Create tool validation tests
   - Implement schema testing suites
   - Setup test fixtures and mocking

2. **Test Coverage Validation Task**
   - Run coverage analysis
   - Identify untested code paths
   - Ensure minimum coverage thresholds
   - Generate coverage reports

3. **Error Scenario Testing Task**
   - Test all identified error conditions
   - Validate error handling and recovery
   - Verify graceful degradation
   - Document error scenarios

### Decision Matrix
| Scenario | Architecture | Implementation Pattern |
|----------|-------------|----------------------|
| Single linear task, few tools | **Monolithic graph** | Simple sequential nodes |
| 2-6 specialized agents, centralized decisions | **Supervisor (tool-calling)** | Supervisor routes to sub-agents |
| >6 agents or multiple domains | **Hierarchical** | Teams with supervisors + coordinator |
| Agents need free communication | **Network** | Many-to-many routing |
| Deterministic pipeline | **Custom workflow** | Explicit edges only |

### Architecture Selection Process
1. Analyze business requirements
2. Count required specialized agents
3. Determine decision-making pattern
4. Assess communication needs
5. Select architecture from matrix
6. Document rationale in graph_spec.yaml

---

## 7. ERROR HANDLING & RECOVERY PATTERNS

### Error Documentation Standard
Every error encountered must be logged in `/docs/tips.md` with:
```markdown
## Error: [Error Type/Message]
**Cause**: [Root cause analysis]
**Solution**: [Step-by-step fix]
**Prevention**: [How to avoid in future]
**Related Files**: [Files affected/modified]
```

### Common Error Patterns & Solutions

#### Router Error Resolution
**Error**: `Expected dict, got <string>`
**Cause**: Router function registered as standard node
**Solution**:
```python
# WRONG - Router as node
builder.add_node("router", router_function)

# CORRECT - Router in conditional edges
builder.add_conditional_edges(
    "source_node",
    router_function,
    {"path1": "target_node1", "path2": "target_node2"}
)
```

#### Import Error Resolution
**Error**: `ImportError: cannot import name 'graph'`
**Cause**: Graph not instantiated in graph.py
**Solution**:
```python
# Add to graph.py
def build_graph():
    # ... graph building logic
    return builder.compile()

# CRITICAL: Instantiate the graph
graph = build_graph()
```

#### Environment Configuration Error
**Error**: Missing API keys
**Solution**:
1. Check `backend/.env` exists
2. Copy to `backend_gen/.env`
3. Validate key with test script
4. Request from user if missing (one time only)

---

## 8. TESTING STRATEGY & VALIDATION

### Four-Tier Testing Approach

#### Tier 0: Unit Testing (Individual Agents & Tools)
- **Agent Unit Tests**: Each agent tested in isolation
- **Tool Validation**: Each tool tested independently
- **Schema Validation**: Pydantic models tested with various inputs
- **Mock Dependencies**: External services mocked for reliable testing
- **Edge Cases**: Error conditions and boundary testing

#### Tier 1: Direct Python Testing
- Import validation
- Graph compilation
- Basic state transitions
- No external dependencies

#### Tier 2: API Integration Testing
- Server startup validation
- Endpoint availability (validated via `curl -N`, not test scripts)
- Request/response cycles
- Error handling

#### Tier 3: End-to-End Testing
- Complete user scenarios
- Multi-turn conversations
- Error recovery
- Performance validation

### Test Execution Protocol
1. **Always start with Tier 0** - Unit test each agent and tool individually
2. **Run Tier 1 only after Tier 0 passes** - Test graph compilation and basic functionality
3. **Execute Tier 2 after Tier 1 succeeds** - Validate API functionality
4. **Run Tier 3 for production readiness** - Complete scenario testing
5. **Re-run Tier 0 & 1 after any code changes** - Ensure no regressions

---

## 9. AUTONOMOUS EXECUTION LOOP

### Task Management Protocol
1. **Task Discovery**: Scan `/tasks` directory for pending tasks
2. **Prerequisite Check**: Verify all dependencies completed
3. **Error Review**: Check `/docs/tips.md` for relevant solutions  
4. **Execution**: Run task with full error handling
5. **Validation**: Verify success criteria met
6. **Status Update**: Mark task as `done` or `failed`
7. **Error Recovery**: On failure, document error, fix, and retry
8. **Continuation**: Move to next task

### Status Tracking System
Tasks marked with status indicators:
- `pending` - Ready for execution
- `in_progress` - Currently executing
- `done` - Successfully completed
- `failed` - Needs attention/retry
- `blocked` - Waiting for dependencies

### Progress Reporting
Maintain `/tasks/progress.md` with:
- Current phase status
- Completed tasks count
- Active issues
- Next scheduled tasks

---


## 11. PRODUCTION DEPLOYMENT CHECKLIST

### Pre-Deployment Validation
- [ ] All unit tests pass (agents, tools, schemas)
- [ ] All integration tests pass (Tier 1, 2, and 3)
- [ ] Test coverage meets minimum requirements (>80%)
- [ ] No errors in server logs
- [ ] Environment variables properly configured
- [ ] Graph compiles and instantiates correctly
- [ ] API endpoints respond correctly (validated via `curl -N`)
- [ ] Error handling covers edge cases

### Deployment Artifacts
- [ ] `langgraph.json` properly configured
- [ ] All dependencies listed and available
- [ ] Environment configuration documented
- [ ] Testing scripts provided
- [ ] Error recovery documentation complete

### Post-Deployment Verification
- [ ] Health check endpoints functional
- [ ] Sample interactions successful
- [ ] Performance within acceptable limits
- [ ] Monitoring and logging operational

---

## 12. APPENDICES

### Appendix A: Complete Example Payloads

#### Basic Interaction Payload
```json
{
  "assistant_id": "agent",
  "input": {
    "messages": [
      {"role": "human", "content": "Enhance this question: What is machine learning?"}
    ]
  },
  "stream_mode": ["updates"]
}
```

#### Streaming Payload
```json
{
  "assistant_id": "agent", 
  "thread_id": "unique-thread-id",
  "input": {
    "messages": [
      {"role": "user", "content": "Please answer the questions in questions.md"}
    ]
  },
  "stream_mode": "values"
}
```

### Appendix B: Server Configuration

#### Default Endpoints
- **API Base**: `http://127.0.0.1:2024`
- **Documentation**: `http://127.0.0.1:2024/docs`
- **Runs Endpoint**: `http://127.0.0.1:2024/runs`
- **Streaming Endpoint**: `http://127.0.0.1:2024/runs/stream`

#### Required Headers
```
Content-Type: application/json
Accept: application/json
```

### Appendix C: Development Tools Configuration

#### Context7 Usage for Documentation
```bash
# Access latest LangGraph documentation
context7 langgraph [specific_topic]
```

#### Essential Commands
```bash
# Development server
langgraph dev

# Installation
pip install -e .

# Testing
python -m pytest tests/ -v

# Graph validation
python -c "from agent.graph import graph; print('Graph loaded successfully')"
```

---


## 10. AGENT USE CASE SPECIFICATION

### Application Description
Create a two-agent LangGraph solution:

**Agent 1: Prompt Enhancer**
- **Input**: Raw user question
- **Function**: Enhance and optimize the question for better AI comprehension
- **Output**: Enhanced, contextually rich prompt
- **Tools**: LLM for prompt engineering, context analysis

**Agent 2: Answer Generator**  
- **Input**: Enhanced prompt from Agent 1
- **Function**: Generate comprehensive, accurate answer
- **Output**: Well-structured response to user
- **Tools**: LLM for answer generation, knowledge retrieval

### Implementation Requirements
1. **State Management**: Track original question, enhanced prompt, and final answer in the graph state, no need for databases.
2. **Error Handling**: Graceful degradation if enhancement fails
3. **Quality Control**: Validation of enhanced prompts and answers
4. **User Experience**: Seamless interaction hiding internal complexity

### Success Criteria
- User submits question → receives enhanced answer
- System handles various question types
- Robust error handling and recovery

---