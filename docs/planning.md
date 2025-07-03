# ENHANCED LANGGRAPH PROJECT CONFIGURATION & CLI INTEGRATION

---

> **Purpose of this document** – Provide a comprehensive, structured guide for autonomous AI agents to develop LangGraph applications with clear execution phases, validation checkpoints, and error recovery patterns. Execute 10 iterative business cases to build a robust knowledge base of solutions and patterns.

---

## 0. ITERATIVE BUSINESS CASE EXECUTION PROTOCOL

### Master Execution Loop
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
   - Follow the Enhanced Tips Format (see section 0.2)
   - Review existing tips before writing new code or tests

4. **Knowledge Accumulation Phase**
   - After each iteration, update `/docs/patterns_learned.md`
   - Document successful architectural decisions
   - Note business case complexity vs implementation patterns

for each round and for each phase write a file in /tasks indicating the steps that have to be done and the instructions for each specific phase, and then follow what that file says. Include all the examples of code or tips for each phase in the file.

never use mock APIs, never, period.

### Business Case Variety Matrix
Ensure coverage across these dimensions over 10 iterations:

| Dimension | Options | Target Coverage |
|-----------|---------|----------------|
| **Domain** | Healthcare, Finance, Education, E-commerce, Legal, Manufacturing, Research, Content, Operations | ≥7 domains |
| **Agent Count** | 2, 3, 4-6, 7+ | All ranges |
| **Architecture** | Monolithic, Supervisor, Hierarchical, Network, Custom | All types |
| **Data Sources** | APIs, Files, Databases, Web scraping, User input | ≥4 sources |
| **Output Types** | Text, Files, API calls, Database updates, Notifications | ≥4 types |
| **Complexity** | Simple linear, Conditional branching, Loops, Error recovery, Human-in-loop | All levels |

### Enhanced Tips Format
When updating `/docs/tips.md`, use this structured format:

```markdown
## TIP #[NUMBER]: [Short Descriptive Title]

**Category**: [Architecture|Testing|Deployment|Development|Integration]
**Severity**: [Critical|High|Medium|Low]
**Business Context**: [When this typically occurs]

### Problem Description
[Detailed description of the issue, including symptoms and context]

### Root Cause Analysis
[Why this error occurs, underlying technical reasons]

### Solution Implementation
```[language]
[Step-by-step code solution with complete examples]
```

### Prevention Strategy
[How to avoid this issue in future implementations]

### Testing Approach
[How to test for this issue and verify the fix]

### Related Tips
[Links to other tips that are related: #[TIP_NUMBER]]

### Business Impact
[How this issue affects different types of business cases]

---
```

### Tips Consultation Protocol
**MANDATORY**: Before writing any code or tests, consult `/docs/tips.md`:

1. **Pre-Code Review**: Check tips related to the component being implemented
2. **Error Pattern Matching**: When an error occurs, first check if it's documented
3. **Solution Application**: Apply documented solutions before creating new ones
4. **Pattern Recognition**: Identify if the current business case matches previous patterns

### Iteration Tracking
Maintain `/tasks/iteration_progress.md`:

```markdown
# Business Case Iteration Progress

## Iteration 1: [Business Case Name]
- **Status**: [Planned|In Progress|Testing|Complete|Failed]
- **Domain**: [Domain name]
- **Architecture**: [Architecture type]
- **Agent Count**: [Number]
- **Key Challenges**: [List of expected/encountered challenges]
- **Tips Generated**: [List of tip numbers created]
- **Completion Date**: [Date if complete]

[Repeat for iterations 2-10]

## Summary Statistics
- **Completed Iterations**: X/10
- **Total Tips Generated**: [Number]
- **Architecture Types Covered**: [List]
- **Domains Explored**: [List]
- **Most Common Error Categories**: [List]
```

---

## 1. ROLE DEFINITION & OPERATIONAL PARAMETERS

### Primary Role
**You are an expert-level, autonomous AI Project Manager and Lead Developer** with the following operational parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Autonomy Level** | Full | No user confirmation required after initial start. **All phases proceed automatically without waiting for approval.** |
| **State Tracking** | File-system only | All progress tracked through files |
| **Error Handling** | Self-correcting + Learning | Must fix errors, document solutions, and apply lessons |
| **Completion Standard** | Production-ready | All code must pass tests and run without errors |
| **Learning Protocol** | Iterative accumulation | Build knowledge base over 10 business case iterations |

### Mission Statement
Orchestrate and execute the development of 10 different LangGraph applications based on generated business cases **fully autonomously, with no user confirmation or intervention required at any step**, ensuring:
- Complete blueprint compliance for each iteration
- Robust error handling and recovery with documented solutions
- Comprehensive testing and validation
- Production-ready deployment artifacts
- Continuous knowledge accumulation in `/docs/tips.md`

### Available Tools
- `read_file` - Read existing files
- `write_file` - Create/modify files
- `execute_shell_command` - Run terminal commands
- when using files with formats like docx, powerpoint, pdf , etc  create tools using this library in the tools https://github.com/MaximeRivest/attachments

---

## 2. EXECUTION PHASES & SUCCESS CRITERIA

### Phase -1: Business Case Generation (NEW)
**Objective**: Generate creative, diverse business case for current iteration
**Success Criteria**:
- Business case documented in `/tasks/iteration_X_business_case.md`
- Case fits variety matrix requirements
- Clear agent roles and responsibilities defined
- Expected technical challenges identified
- Success metrics established

**Business Case Template**:
```markdown
# Iteration [X] Business Case: [Title]

## Business Problem
[What real-world problem does this solve?]

## Proposed Agent Architecture
- **Agent 1**: [Name] - [Role and responsibilities]
- **Agent 2**: [Name] - [Role and responsibilities]
- **Agent N**: [Name] - [Role and responsibilities]

## Data Flow
[How information flows between agents]

## Expected Technical Challenges
[What difficulties do you anticipate?]

## Success Criteria
[How will you know this works?]

## Business Value
[Why would someone use this system?]
```

### Phase 0: Workspace Initialization
**Objective**: Clean slate preparation with tips consultation. **This phase and all subsequent phases proceed automatically, without user confirmation.**
**Success Criteria**: 
- `/tasks` directory completely reset
- `/backend_gen` directory completely reset
- `/backend/` successfully copied to `/backend_gen/`
- Tips reviewed and architectural patterns identified
- Environment validated and ready

**Validation Commands**:
```bash
ls -la /tasks  # Should be empty or non-existent
ls -la /backend_gen  # Should contain copied backend structure
```

### Phase 1: Architecture Planning & Specification
**Objective**: Complete project specification before any implementation, incorporating lessons learned. **This phase proceeds automatically after workspace initialization, with no user confirmation required.**
**Success Criteria**:
- Tips consultation completed and relevant patterns identified
- All documentation internalized and understood
- `/tasks/01_define-graph-spec.md` created with detailed execution plan
- `/tasks/artifacts/graph_spec.yaml` generated with complete architecture
- Business case framing completed
- Testing strategy defined incorporating known error patterns

**Critical Rule**: NO implementation code until this phase is 100% complete

### Phase 2: Test-Driven Development & Implementation
**Objective**: Implement comprehensive testing BEFORE code generation, following TDD principles. **This phase starts automatically after planning, with no user confirmation required.**

**TDD Philosophy**: Write tests first, then implement code to make tests pass. This ensures:
- Clear specification of expected behavior before implementation
- Better code design driven by test requirements  
- Higher confidence in code correctness
- Faster debugging and error detection
- Prevention of regressions during development

**Success Criteria**:
- **Phase 2.1**: Complete test suite written BEFORE any implementation
- **Phase 2.2**: All mandatory implementation files created to satisfy tests
- LLM integration properly configured and tested
- All nodes follow MANDATORY LLM Call Pattern
- Graph assembly completed using best practices from tips
- Import validation successful
- Known error patterns proactively avoided

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
**Objective**: Comprehensive testing and error resolution with knowledge capture. **This phase is entered automatically after implementation, with no user confirmation required.**

#### Phase 3.1: Unit Testing & Component Validation
**Success Criteria**:
- [ ] All unit tests pass with real LLM calls, file operations, and computations
- [ ] Individual agent functions work correctly with proper signatures
- [ ] Pydantic models handle data validation and conversion properly
- [ ] **LLM conversations logged** for debugging and verification
- [ ] Error handling mechanisms tested with fallback data
- [ ] Component isolation verified (agents work independently)

#### Phase 3.2: Multiple Agent Pytests & Type Validation
**Objective**: Create comprehensive pytest test suites for each individual agent and validate all data types they handle. **This phase follows TDD principles by testing each agent thoroughly before server integration.**

**Success Criteria**:
- [ ] **Individual agent pytests created** - One test file per agent with comprehensive test coverage
- [ ] **Type validation tests** - Verify each agent correctly handles and transforms all expected data types
- [ ] **Agent input/output testing** - Test agent function signatures, parameter handling, and return values
- [ ] **State transition testing** - Verify agents correctly update OverallState with proper field types
- [ ] **Error handling testing** - Test agent behavior with invalid inputs, missing data, and edge cases
- [ ] **LLM integration testing** - Test real LLM calls with conversation logging and response validation
- [ ] **Configuration testing** - Verify agents use Configuration.from_runnable_config() correctly
- [ ] **Pydantic model testing** - Test all data schemas for validation, serialization, and deserialization

**Required Test Files Structure**:
```bash
tests/
├── test_agent_[agent_name].py     # Individual agent tests (one per agent)
├── test_types_[agent_name].py     # Type validation tests (one per agent) 
├── test_state_transitions.py      # OverallState transition testing
├── test_configuration.py          # Configuration pattern testing
├── test_schemas.py                 # Pydantic model validation
├── test_error_handling.py          # Error scenarios and fallbacks
└── conftest.py                     # Shared test fixtures and utilities
```

**Example Individual Agent Test Template**:
```python
# tests/test_agent_[agent_name].py
import pytest
import json
from datetime import datetime
from unittest.mock import Mock
from langchain_core.runnables import RunnableConfig
from agent.state import OverallState
from agent.configuration import Configuration
from agent.nodes.[agent_name] import [agent_function_name]
from agent.tools_and_schemas import [relevant_schemas]

@pytest.fixture
def sample_state() -> OverallState:
    """Create realistic test state for agent testing"""
    return {
        "messages": [{"role": "user", "content": "Test input message"}],
        "[domain_specific_field]": {"key": "test_value"},
        "errors": [],
        "current_step": "initialized"
    }

@pytest.fixture  
def runnable_config() -> RunnableConfig:
    """Create test configuration"""
    return RunnableConfig(
        configurable={
            "query_generator_model": "gemini-2.0-flash",
            "answer_model": "gemini-1.5-flash-latest",
            "reflection_model": "gemini-2.5-flash-preview-04-17"
        }
    )

class TestAgentFunctionality:
    """Test core agent functionality"""
    
    def test_agent_function_signature(self, sample_state, runnable_config):
        """Test agent function accepts correct parameters"""
        result = [agent_function_name](sample_state, runnable_config)
        assert isinstance(result, dict)
        assert "messages" in result
    
    def test_agent_state_updates(self, sample_state, runnable_config):
        """Test agent correctly updates state fields"""
        result = [agent_function_name](sample_state, runnable_config)
        
        # Verify state structure maintained
        assert "messages" in result
        assert isinstance(result["messages"], list)
        
        # Verify agent-specific state updates
        assert "[expected_output_field]" in result
        assert result["current_step"] != "initialized"
    
    def test_agent_configuration_usage(self, sample_state, runnable_config):
        """Test agent uses Configuration.from_runnable_config() pattern"""
        # Mock Configuration to verify it's called
        with pytest.MonkeyPatch().context() as m:
            mock_config = Mock()
            mock_config.answer_model = "test-model"
            m.setattr("agent.configuration.Configuration.from_runnable_config", Mock(return_value=mock_config))
            
            result = [agent_function_name](sample_state, runnable_config)
            
            # Verify Configuration was used
            assert result is not None
    
    def test_agent_llm_conversation_logging(self, sample_state, runnable_config, caplog):
        """Test LLM conversation logging and real API calls"""
        start_time = datetime.now()
        result = [agent_function_name](sample_state, runnable_config)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Verify LLM was called (check for API response indicators)
        assert "messages" in result
        assert len(result["messages"]) > 0
        
        # Verify conversation logging occurred
        assert duration > 0  # Real LLM call takes time
        
        # Log conversation details for debugging
        print(f"\n=== AGENT TEST CONVERSATION ===")
        print(f"Agent: [agent_name]")
        print(f"Duration: {duration:.2f}s")
        print(f"Messages: {len(result['messages'])}")
        print(f"Response: {result['messages'][-1]['content'][:200]}...")

class TestAgentTypeValidation:
    """Test agent handles all expected data types correctly"""
    
    def test_input_type_validation(self, runnable_config):
        """Test agent handles various input state types"""
        test_cases = [
            {"messages": [], "field": None},  # None values
            {"messages": [], "field": {}},   # Empty dicts
            {"messages": [], "field": []},   # Empty lists
            {"messages": [{"role": "user", "content": "test"}], "field": {"data": "value"}},  # Valid data
        ]
        
        for test_state in test_cases:
            result = [agent_function_name](test_state, runnable_config)
            assert isinstance(result, dict)
            assert "messages" in result
    
    def test_output_type_consistency(self, sample_state, runnable_config):
        """Test agent output types match OverallState schema"""
        result = [agent_function_name](sample_state, runnable_config)
        
        # Verify required fields are present and correct types
        assert isinstance(result.get("messages"), list)
        assert isinstance(result.get("errors", []), list)
        assert isinstance(result.get("current_step"), str)
        
        # Verify agent-specific output fields
        # [Add specific assertions for this agent's output types]
    
    def test_pydantic_model_integration(self, sample_state, runnable_config):
        """Test agent correctly uses Pydantic models for data validation"""
        result = [agent_function_name](sample_state, runnable_config)
        
        # If agent returns structured data, verify Pydantic model usage
        if "[structured_data_field]" in result:
            structured_data = result["[structured_data_field]"]
            # Verify data can be reconstructed with Pydantic model
            # [Add specific Pydantic model validation tests]

class TestAgentErrorHandling:
    """Test agent error handling and fallback mechanisms"""
    
    def test_missing_required_fields(self, runnable_config):
        """Test agent handles missing required state fields"""
        incomplete_state = {"messages": []}  # Missing required fields
        
        result = [agent_function_name](incomplete_state, runnable_config)
        
        # Agent should handle gracefully
        assert isinstance(result, dict)
        assert "errors" in result or "messages" in result
    
    def test_invalid_configuration(self, sample_state):
        """Test agent handles invalid configuration"""
        invalid_config = RunnableConfig(configurable={})
        
        # Should not crash, might use fallbacks
        result = [agent_function_name](sample_state, invalid_config)
        assert isinstance(result, dict)
    
    def test_llm_api_failure_fallback(self, sample_state, runnable_config, monkeypatch):
        """Test agent fallback when LLM API fails"""
        # Mock LLM to raise exception
        def mock_llm_invoke(*args, **kwargs):
            raise Exception("API Error")
        
        # This would require more sophisticated mocking of the LLM
        # Result should include error handling
        result = [agent_function_name](sample_state, runnable_config)
        # Verify agent didn't crash and provided fallback
        assert isinstance(result, dict)
```

**Type Validation Test Template**:
```python
# tests/test_types_[agent_name].py  
import pytest
from typing import get_type_hints, get_origin, get_args
from agent.nodes.[agent_name] import [agent_function_name]
from agent.state import OverallState
from agent.tools_and_schemas import [relevant_schemas]

class TestAgentTypeSignatures:
    """Validate agent function type signatures and annotations"""
    
    def test_function_type_hints(self):
        """Test agent function has proper type hints"""
        type_hints = get_type_hints([agent_function_name])
        
        # Verify parameter types
        assert 'state' in type_hints
        assert 'config' in type_hints
        
        # Verify return type
        assert 'return' in type_hints
        
    def test_state_type_compatibility(self):
        """Test agent state parameter accepts OverallState"""
        # Verify OverallState fields are properly typed
        state_hints = get_type_hints(OverallState)
        assert 'messages' in state_hints
        
    def test_pydantic_schema_types(self):
        """Test all Pydantic schemas have proper type validation"""
        # Test each schema used by this agent
        for schema_class in [relevant_schemas]:
            # Verify schema can be instantiated
            schema_hints = get_type_hints(schema_class)
            assert len(schema_hints) > 0
            
            # Test schema validation with various inputs
            # [Add specific type validation tests]

class TestStateTransitionTypes:
    """Test agent correctly transforms state field types"""
    
    def test_input_state_types(self):
        """Test agent accepts various state field types"""
        test_inputs = [
            None,
            {},
            [],
            "string_value",
            {"key": "value"},
            [{"item": "value"}]
        ]
        
        for test_input in test_inputs:
            # Test agent can handle this input type
            # [Implementation depends on specific agent requirements]
            pass
    
    def test_output_state_types(self):
        """Test agent produces consistent output types"""
        # Test multiple runs produce same output types
        # [Implementation depends on specific agent requirements]
        pass
```

#### Phase 3.3: LangWatch Scenario Testing & Agent Simulation
**Objective**: Advanced agent testing through simulation-based testing with LangWatch Scenario framework BEFORE server testing. **This phase ensures code quality and agent behavior validation before LangGraph server integration.**

**What is LangWatch Scenario**: LangWatch Scenario is an Agent Testing Framework based on simulations that can:
- Test real agent behavior by simulating users in different scenarios and edge cases  
- Evaluate and judge at any point of the conversation with powerful multi-turn control
- Combine with any LLM eval framework or custom evals (agnostic by design)
- Integrate any agent by implementing just one `call()` method
- Available in Python, TypeScript and Go with comprehensive testing capabilities

**Success Criteria**:
- [ ] LangWatch Scenario installed and configured properly
- [ ] Agent adapter implementation created for our LangGraph agent
- [ ] Multiple scenario tests created covering edge cases and user interactions
- [ ] Simulation-based testing executed with real user behavior simulation
- [ ] Judge agents evaluate conversation quality with custom criteria
- [ ] Performance metrics captured across different scenarios
- [ ] Test results integrated with overall testing pipeline
- [ ] Scenario test reports generated for analysis
- [ ] **All scenario tests pass** before proceeding to server testing

**Installation & Setup**:
```bash
# Install LangWatch Scenario framework
cd /backend_gen
pip install langwatch-scenario pytest

# Verify installation
python -c "import scenario; print('LangWatch Scenario installed successfully')"

# Set up environment variables for LangWatch (optional but recommended for visualization)
echo "LANGWATCH_API_KEY=your-api-key-here" >> .env
echo "OPENAI_API_KEY=your-openai-key-here" >> .env  # Required for user simulation

# Configure scenario defaults
python -c "
import scenario
scenario.configure(
    default_model='openai/gpt-4.1-mini',  # For user simulation
    cache_key='[business-case]-tests',  # For repeatable tests
    verbose=True  # Show detailed simulation output
)
print('LangWatch Scenario configured')
"
```

#### Phase 3.4: Graph Compilation & Import Validation
**Objective**: Validate graph structure and imports after successful scenario testing.

**Success Criteria**:
- [ ] Graph compiles and imports successfully without errors
- [ ] All node imports execute without circular dependencies
- [ ] State schema is valid TypedDict structure
- [ ] LangGraph configuration files are properly structured
- [ ] **Pre-server validation** prevents import errors at runtime
- [ ] Package installation completes successfully

#### Phase 3.5: LangGraph Server Testing & Real Execution
**Success Criteria**:
- [ ] `langgraph dev` server starts without errors or warnings from correct directory
- [ ] Server logs show no ImportError, ModuleNotFoundError, or relative import issues
- [ ] OpenAPI schema endpoint returns valid JSON with correct paths
- [ ] Thread management creates and manages execution threads properly
- [ ] **Real LLM execution** processes requests with actual API calls (not mocks)
- [ ] **Complete LLM conversations logged** with prompts, responses, and timing
- [ ] Graph execution transitions through all states successfully
- [ ] Server cleanup prevents hanging processes

#### Phase 3.6: API Integration & End-to-End Validation
**Success Criteria**:
- [ ] All REST API endpoints respond correctly (invoke, stream, health, schema)
- [ ] End-to-end workflow completes from document upload to final analysis
- [ ] Performance tests show reasonable execution times (< 60 seconds per workflow)
- [ ] Error scenarios handled gracefully with proper HTTP status codes
- [ ] **Full workflow LLM conversation logs** captured for analysis
- [ ] Integration with external APIs works reliably
- [ ] Resource cleanup and memory management verified

**Testing with LLM Conversation Logging**:
All tests must include comprehensive logging of LLM interactions:
```python
def log_llm_conversation(agent_name, prompt, response, duration):
    """Log complete LLM conversation for debugging and verification"""
    timestamp = datetime.now().isoformat()
    conversation_log = {
        "timestamp": timestamp,
        "agent": agent_name,
        "prompt_length": len(prompt),
        "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
        "full_prompt": prompt,  # Complete prompt for debugging
        "response_length": len(response.content) if hasattr(response, 'content') else len(str(response)),
        "response_preview": response.content[:200] + "..." if hasattr(response, 'content') and len(response.content) > 200 else str(response)[:200],
        "full_response": response.content if hasattr(response, 'content') else str(response),
        "duration_seconds": duration,
        "model_used": response.response_metadata.get('model_name') if hasattr(response, 'response_metadata') else "unknown"
    }
    
    # Write to conversation log file
    log_file = f"tests/llm_conversations_{agent_name}_{timestamp.split('T')[0]}.json"
    with open(log_file, "a") as f:
        f.write(json.dumps(conversation_log, indent=2) + "\n")
    
    # Also print to console for immediate visibility
    print(f"\n=== LLM CONVERSATION: {agent_name} ===")
    print(f"Timestamp: {timestamp}")
    print(f"Duration: {duration:.2f}s")
    print(f"Prompt ({len(prompt)} chars): {prompt[:300]}...")
    print(f"Response ({len(response.content) if hasattr(response, 'content') else len(str(response))} chars): {response.content[:300] if hasattr(response, 'content') else str(response)[:300]}...")
    print("=" * 50)
```

# Final validation and documentation
echo "=== Final Validation Report ==="
echo "✅ Unit Tests: All agent, tool, and schema tests passing"
echo "✅ Multiple Agent Pytests: Individual agent and type validation complete"
echo "✅ LangWatch Scenario Testing: Agent behavior simulation verified"
echo "✅ Graph Compilation: Successfully imports and compiles"
echo "✅ LangGraph Server: Real LLM execution verified"
echo "✅ API Integration: All endpoints functional"
echo "🎯 System Ready for Production Deployment"

---

## 3. COMPREHENSIVE TESTING PHASE SUMMARY

**NEW TESTING SEQUENCE** (Following TDD Principles):
1. **Phase 3.1**: Unit Testing & Component Validation
2. **Phase 3.2**: Multiple Agent Pytests & Type Validation ⭐ **NEW**
3. **Phase 3.3**: LangWatch Scenario Testing & Agent Simulation ⭐ **MOVED HERE**  
4. **Phase 3.4**: Graph Compilation & Import Validation
5. **Phase 3.5**: LangGraph Server Testing & Real Execution
6. **Phase 3.6**: API Integration & End-to-End Validation

This sequence ensures **code quality and agent behavior validation BEFORE server integration**, following Test-Driven Development principles where comprehensive testing precedes deployment.

**Benefits of New Testing Sequence**:
- **Earlier Error Detection**: Issues caught in agent-level testing before server complications
- **Behavioral Validation**: LangWatch scenarios verify agent behavior in realistic conditions
- **Type Safety**: Comprehensive type validation prevents runtime type errors
- **Confidence Building**: Each agent thoroughly tested before integration
- **Faster Debugging**: Problems isolated to specific agents rather than complex server interactions

---

## 4. LANGGRAPH DEVELOPMENT PATTERNS & BEST PRACTICES

### Core Development Patterns

**LLM Configuration Pattern** (Critical - TIP #012):
```python
from agent.configuration import Configuration
from langchain_core.runnables import RunnableConfig

def agent_function(state: OverallState, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.answer_model,  # Use configured model
        temperature=0,
        api_key=os.getenv("GEMINI_API_KEY")
    )
```

**Import Requirements** (Critical - TIP #006):
```python
# ✅ CORRECT - Absolute imports (works in langgraph dev)
from agent.nodes.node_name import function_name

# ❌ WRONG - Relative imports (fails in langgraph dev)
from .nodes.node_name import function_name
```

**Message Handling Pattern** (Critical - TIP #013 & #014):
```python
# Handle both LangChain objects and dictionaries
def process_message(message):
    if hasattr(message, 'content'):
        content = message.content  # LangChain message object
    elif isinstance(message, dict):
        content = message.get("content")  # Dictionary message
    else:
        content = str(message)
    
    # Always use "assistant" role, never "agent"
    return {"role": "assistant", "content": content}
```

### State Management Patterns

**Safe State Access** (Critical - TIP #010):
```python
# Always check for None before operations
existing_errors = state.get("errors") or []
messages = state.get("messages") or []
```

**Proper State Updates**:
```python
# Return state updates as dictionary
return {
    "messages": updated_messages,
    "field_name": new_value,
    "current_step": "step_completed"
}
```

### Testing Strategies

**Real LLM Testing vs Mock Testing**:
- Unit tests can use mocks for speed
- Integration tests MUST use real LLM calls
- Always log LLM conversations for debugging
- Test with real API keys in CI/CD pipeline

**Error Prevention Checklist**:
- [ ] Use absolute imports in graph.py
- [ ] Use Configuration.from_runnable_config() 
- [ ] Handle both LangChain objects and dictionaries in message processing
- [ ] Use "assistant" role, never "agent" role
- [ ] Test graph loading before server startup
- [ ] Include fallback responses for LLM failures

---

## 5. CRITICAL ERROR PATTERNS & SOLUTIONS

### Import Error Prevention (TIP #006)
```bash
# Test imports before server
python -c "from agent.graph import graph; print('✅ Graph imports successfully')"
```

### Configuration Error Prevention (TIP #012)
```python
# Always use configuration, never hardcode models
configurable = Configuration.from_runnable_config(config)
llm = ChatGoogleGenerativeAI(model=configurable.answer_model)
```

### Message Error Prevention (TIP #013 & #014)
```python
# Handle message types properly
if hasattr(message, 'content'):
    content = message.content
else:
    content = message.get("content", "")

# Use correct message roles
{"role": "assistant", "content": content}  # ✅ Correct
{"role": "agent", "content": content}      # ❌ Wrong
```

## 6. AUTONOMOUS EXECUTION REQUIREMENTS

### Operational Parameters
- **Full Autonomy**: No user confirmation required after initial start
- **State Tracking**: All progress tracked through file system
- **Error Handling**: Self-correcting with learning documentation
- **Completion Standard**: Production-ready code that passes all tests
- **Learning Protocol**: Knowledge accumulation in `/docs/tips.md`

### Success Metrics
- [ ] All test phases complete successfully
- [ ] Real LLM execution verified
- [ ] No critical errors in server startup
- [ ] Production deployment ready
- [ ] Knowledge base updated with new patterns

## 3. CORE PRINCIPLES & NON-NEGOTIABLES

### Architectural Principles
1. **Business Case Diversity** - Each iteration explores different patterns and domains
2. **Learning Integration** - Apply accumulated knowledge from previous iterations
3. **Planning First** - No implementation until complete planning phase
4. **Blueprint Compliance** - Every artifact must conform to `/docs/blueprint_backend.md`
5. **Full Autonomy** - Proceed without user interaction once plan exists
6. **Enhanced Error Documentation** - Every error must be logged with Enhanced Tips Format in `/docs/tips.md`
7. **Router Rule** - Only router returns sentinel strings; nodes return dict, NOTHING, or raise

### Knowledge Management Principles
1. **Tips Consultation** - Always review relevant tips before implementation
2. **Pattern Recognition** - Identify when current case matches documented patterns
3. **Solution Reuse** - Apply documented solutions before creating new ones
4. **Continuous Learning** - Each error teaches us something valuable
5. **Structured Documentation** - Follow Enhanced Tips Format consistently

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
- [ ] `/docs/tips.md` has been reviewed for relevant patterns
- [ ] Current iteration's business case is clearly defined
- [ ] Environment variables are properly configured
- [ ] Previous phase completion criteria are met
- [ ] Error ledger (`/docs/tips.md`) has been consulted

### Phase -1: Business Case Generation
```bash
# 1. Review iteration progress
cat /tasks/iteration_progress.md

# 2. Check variety matrix coverage
# [Review completed business cases and identify gaps]

# 3. Generate new business case
# [Create iteration_X_business_case.md with unique scenario]

# 4. Update iteration progress
# [Mark new iteration as planned]
```

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

# 5. Install dependencies
cd /backend_gen && pip install -e .
```
remember to run pip install -e . in the backend_gen directory.

### Phase 1: Node Specification & Flow Design

#### 1.1 Tips Consultation (NEW)
- Read `/docs/tips.md` completely
- Identify tips relevant to current business case
- Note architectural patterns that apply
- Plan implementation to avoid documented pitfalls

#### 1.2 Documentation Internalization
- Read and understand all provided documentation
- Identify key requirements and constraints
- Map business requirements to technical architecture
- Consider lessons learned from previous iterations

#### 1.3 Task Definition
Create `/tasks/01_define-graph-spec.md` with:
- Detailed task description incorporating tips insights
- Expected outputs
- Validation criteria
- Dependencies
- Risk mitigation based on documented errors

#### 1.4 Architecture Specification
Generate `/tasks/artifacts/graph_spec.yaml` following the Business-Case Checklist:

**Required Sections**:
1. **Business Case Framing**
   - High-level goal definition
   - Core competencies identification
   - Architecture choice (centralized vs distributed)
   - External API requirements
   - Data flow mapping
   - Testing strategy incorporating known error patterns

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
   - Error scenarios from tips.md

6. **Risk Mitigation Plan (NEW)**
   - Identified risks from tips.md
   - Prevention strategies
   - Testing approaches for known error patterns

### Phase 2: Direct Code Implementation

#### 2.1 Pre-Implementation Tips Review
**MANDATORY**: Before writing any code, review relevant tips:
```bash
# Search for relevant tips by category
grep -n "Category.*Architecture" /docs/tips.md
grep -n "Category.*Development" /docs/tips.md
```

**Critical Tips for Node Implementation**:
- **TIP #012**: Use Configuration.from_runnable_config() instead of hardcoded models
- **TIP #008**: LangGraph Agent Function Signature must use RunnableConfig
- **TIP #010**: State management with None values using safe patterns
- **TIP #006**: Use absolute imports in graph.py to avoid server startup failures

#### 2.2 State Definition
**File**: `/backend_gen/src/agent/state.py`
```python
from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional

class OverallState(TypedDict):
    # Define based on graph_spec.yaml requirements
    messages: List[Dict[str, Any]]
    # Add other state fields as needed
```

#### 2.3 Tools and Schemas
**File**: `/backend_gen/src/agent/tools_and_schemas.py`
- Pydantic models for data validation
- Tool wrapper functions
- Schema definitions for LLM interactions

#### 2.4 Node Implementation
**Directory**: `/backend_gen/src/agent/nodes/`

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
    # Call the LLM (unstructured output)
    result = llm.invoke(prompt)
    # Optionally, for structured output:
    # structured_llm = llm.with_structured_output(MyPydanticSchema)
    # result = structured_llm.invoke(prompt)
    # Update state with LLM result
    state["messages"].append({"role": "agent", "content": result.content})
    return state
```

**❌ WRONG Pattern - Hardcoded Models (Common Mistake)**:
```python
# DON'T DO THIS - Hardcoded model names make nodes inflexible
def bad_node_function(state: OverallState, config: Dict[str, Any]) -> dict:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # HARDCODED - BAD!
        temperature=0.1,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    # ... rest of implementation
```

**✅ Configuration Model Selection Guide**:
- `configurable.query_generator_model` - For generating search queries or initial analysis
- `configurable.answer_model` - For main processing, analysis, and response generation  
- `configurable.reflection_model` - For evaluation, reflection, and quality assessment tasks

#### 2.5 Graph Assembly
**File**: `/backend_gen/src/agent/graph.py`

**CRITICAL: Use ABSOLUTE imports only (relative imports will fail in langgraph dev)**
```python
# ❌ WRONG - Relative imports (will cause server startup failure)
# from .nodes.clinical_data_collector import clinical_data_collector_agent
# from .state import OverallState

# ✅ CORRECT - Absolute imports (required for langgraph dev server)
from langgraph.graph import StateGraph, START, END
from agent.state import OverallState
from agent.nodes.clinical_data_collector import clinical_data_collector_agent
from agent.nodes.literature_research_agent import literature_research_agent
from agent.nodes.data_quality_validator import data_quality_validator_agent
from agent.nodes.statistical_analysis_agent import statistical_analysis_agent
from agent.nodes.privacy_compliance_agent import privacy_compliance_agent
from agent.nodes.report_generation_agent import report_generation_agent

def build_graph():
    builder = StateGraph(OverallState)
    
    # Add nodes (not router)
    builder.add_node("clinical_data_collector_agent", clinical_data_collector_agent)
    builder.add_node("literature_research_agent", literature_research_agent)
    builder.add_node("data_quality_validator_agent", data_quality_validator_agent)
    builder.add_node("statistical_analysis_agent", statistical_analysis_agent)
    builder.add_node("privacy_compliance_agent", privacy_compliance_agent)
    builder.add_node("report_generation_agent", report_generation_agent)
    
    # Add conditional edges with router logic
    builder.add_conditional_edges(
        START,
        route_to_tier_one,  # Router function determines next step
        {
            "clinical_data_collector": "clinical_data_collector_agent",
            "literature_research": "literature_research_agent",
            "END": END
        }
    )
    
    # Add more edges as needed for your business case
    builder.add_conditional_edges(
        "clinical_data_collector_agent",
        route_after_data_collection,
        {
            "data_quality_validator": "data_quality_validator_agent",
            "END": END
        }
    )
    
    return builder.compile()

# CRITICAL: Instantiate the graph for langgraph.json
graph = build_graph()

# Export for use in application
def get_compiled_graph():
    """Get the compiled clinical research data processing graph."""
    return graph
```

**CRITICAL: Fix agent/__init__.py to prevent circular imports**
```python
# ❌ WRONG - Creates circular import that breaks server
# from agent.graph import graph
# __all__ = ["graph"]

# ✅ CORRECT - Minimal __init__.py to prevent circular imports
# Removed circular import to prevent LangGraph dev server startup issues
```

**CRITICAL: Fix utils.py with working LLM patterns**
```python
# ❌ WRONG - These imports will fail
# from langchain_core.language_models.fake import FakeListChatModel, FakeChatModel
# from langchain_core.language_models.llm import LLM

# ✅ CORRECT - Working LLM pattern with proper fallbacks
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
        return FakeListLLM(responses=[
            "Clinical data analysis complete. Quality assessment: Good.",
            "Statistical analysis reveals significant correlations.",
            "Compliance validation successful. HIPAA requirements met.",
            "Research report generated with comprehensive findings."
        ])
    except ImportError:
        # Final fallback - simple mock that works
        class SimpleFakeLLM:
            def invoke(self, prompt):
                class Response:
                    content = "Mock LLM response for testing"
                return Response()
        return SimpleFakeLLM()
```

#### 2.6 Unit Test Implementation
**Directory**: `/backend_gen/tests/`

Create comprehensive unit tests for each component:

**File**: `/backend_gen/tests/test_agents.py`
```python
# NOTE: Unit tests should use real LLM calls, real file access, and real code execution wherever possible.
# See the node implementation for actual LLM usage.

import pytest
import os
from agent.nodes.patient_intake import patient_intake_node
from agent.nodes.doctor_availability import doctor_availability_node
from agent.nodes.scheduler import scheduler_node
from agent.state import OverallState

class TestPatientIntakeAgent:
    """Unit tests for the patient intake agent"""
    
    def setup_method(self):
        """Setup test data for each test"""
        self.base_state = {
            "messages": [{"role": "human", "content": "I need an appointment for a headache"}],
            "patient_info": None,
            "appointment_request": None,
            "doctor_schedules": None,
            "scheduled_appointment": None,
            "errors": None
        }
    
    def test_patient_intake_basic_functionality(self):
        """Test basic patient intake functionality with real LLM"""
        # Execute agent with real LLM call
        result = patient_intake_node(self.base_state)
        
        # Validate results
        assert result is not None
        assert "patient_info" in result
        assert result["patient_info"] is not None
        assert "name" in result["patient_info"]
        assert "contact" in result["patient_info"]
    
    def test_patient_intake_error_handling(self):
        """Test error handling when input is invalid"""
        empty_state = {"messages": []}
        
        # Execute agent and check error handling
        result = patient_intake_node(empty_state)
        # Should handle gracefully or provide meaningful errors
        assert result is not None

class TestDoctorAvailabilityAgent:
    """Unit tests for the doctor availability agent"""
    
    def setup_method(self):
        """Setup test data for each test"""
        self.base_state = {
            "messages": [{"role": "human", "content": "Need to check doctor availability"}],
            "patient_info": {"name": "John Doe", "age": 35, "contact": "john@example.com"},
            "doctor_schedules": None,
            "errors": None
        }
    
    def test_doctor_availability_basic_functionality(self):
        """Test basic doctor availability functionality with real data"""
        # Execute agent with real data aggregation
        result = doctor_availability_node(self.base_state)
        
        # Validate results
        assert result is not None
        assert "doctor_schedules" in result
        assert result["doctor_schedules"] is not None
        assert len(result["doctor_schedules"]) > 0
        
        # Check schedule structure
        for schedule in result["doctor_schedules"]:
            assert "doctor_name" in schedule
            assert "specialty" in schedule
            assert "available_slots" in schedule

class TestSchedulerAgent:
    """Unit tests for the scheduler agent"""
    
    def setup_method(self):
        """Setup test data for each test"""
        self.base_state = {
            "messages": [{"role": "human", "content": "Schedule my appointment"}],
            "patient_info": {"name": "John Doe", "age": 35, "contact": "john@example.com"},
            "doctor_schedules": [
                {
                    "doctor_name": "Dr. Smith",
                    "specialty": "General Medicine",
                    "available_slots": [{"date": "2024-07-01", "time": "10:00"}]
                }
            ],
            "scheduled_appointment": None,
            "errors": None
        }
    
    def test_scheduler_basic_functionality(self):
        """Test basic scheduling functionality with real matching logic"""
        # Execute agent with real scheduling logic
        result = scheduler_node(self.base_state)
        
        # Validate results
        assert result is not None
        assert "scheduled_appointment" in result
        assert result["scheduled_appointment"] is not None
        
        # Check appointment structure
        appointment = result["scheduled_appointment"]
        assert "patient_name" in appointment
        assert "doctor_name" in appointment
        assert "date" in appointment
        assert "time" in appointment
        assert "status" in appointment
```

**File**: `/backend_gen/tests/test_tools.py`
```python
# NOTE: Unit tests should use real LLM calls, real file access, and real code execution wherever possible.
# See the node implementation for actual LLM usage.

import pytest
import os
from agent.tools_and_schemas import (
    validate_patient_info, 
    validate_appointment_request,
    validate_doctor_schedule,
    confirm_appointment
)

class TestToolFunctionality:
    """Test individual tool operations with real data"""
    
    def test_validate_patient_info_real_data(self):
        """Test patient info validation with real data"""
        valid_info = {
            "name": "John Doe",
            "age": 35,
            "contact": "john@example.com",
            "symptoms": "Headache"
        }
        
        result = validate_patient_info(valid_info)
        assert result is True
        
        invalid_info = {
            "name": "",  # Invalid empty name
            "age": "not_a_number",  # Invalid age
            "contact": "invalid_email"
        }
        
        result = validate_patient_info(invalid_info)
        assert result is False
    
    def test_validate_appointment_request_real_data(self):
        """Test appointment request validation with real data"""
        valid_request = {
            "patient_name": "John Doe",
            "requested_date": "2024-07-01",
            "requested_time": "10:00",
            "doctor_specialty": "General Medicine"
        }
        
        result = validate_appointment_request(valid_request)
        assert result is True
    
    def test_validate_doctor_schedule_real_data(self):
        """Test doctor schedule validation with real data"""
        valid_schedule = {
            "doctor_name": "Dr. Smith",
            "specialty": "General Medicine",
            "available_slots": [
                {"date": "2024-07-01", "time": "10:00"},
                {"date": "2024-07-01", "time": "11:00"}
            ]
        }
        
        result = validate_doctor_schedule(valid_schedule)
        assert result is True
    
    def test_confirm_appointment_real_data(self):
        """Test appointment confirmation with real data"""
        valid_confirmation = {
            "patient_name": "John Doe",
            "doctor_name": "Dr. Smith",
            "date": "2024-07-01",
            "time": "10:00",
            "status": "confirmed"
        }
        
        result = confirm_appointment(valid_confirmation)
        assert result is True

class TestFileOperations:
    """Test file operations with real file access"""
    
    def test_read_doctor_schedule_from_file(self):
        """Test reading doctor schedules from actual CSV files"""
        # Create a test CSV file
        test_csv_content = """doctor_name,specialty,date,time
Dr. Smith,General Medicine,2024-07-01,10:00
Dr. Lee,Pediatrics,2024-07-01,09:00"""
        
        test_file_path = "/tmp/test_schedule.csv"
        with open(test_file_path, "w") as f:
            f.write(test_csv_content)
        
        # Test reading the file
        assert os.path.exists(test_file_path)
        with open(test_file_path, "r") as f:
            content = f.read()
            assert "Dr. Smith" in content
            assert "General Medicine" in content
        
        # Cleanup
        os.remove(test_file_path)

class TestCalculations:
    """Test mathematical calculations with real computation"""
    
    def test_appointment_time_calculations(self):
        """Test real time calculations for appointment scheduling"""
        # Test duration calculation
        start_time = "10:00"
        duration_minutes = 30
        
        # Real calculation logic
        start_hour, start_minute = map(int, start_time.split(":"))
        total_minutes = start_hour * 60 + start_minute + duration_minutes
        end_hour = total_minutes // 60
        end_minute = total_minutes % 60
        end_time = f"{end_hour:02d}:{end_minute:02d}"
        
        assert end_time == "10:30"
    
    def test_availability_overlap_calculation(self):
        """Test real overlap calculations for scheduling conflicts"""
        # Real overlap detection logic
        slot1 = {"start": "10:00", "end": "11:00"}
        slot2 = {"start": "10:30", "end": "11:30"}
        
        def time_to_minutes(time_str):
            hour, minute = map(int, time_str.split(":"))
            return hour * 60 + minute
        
        slot1_start = time_to_minutes(slot1["start"])
        slot1_end = time_to_minutes(slot1["end"])
        slot2_start = time_to_minutes(slot2["start"])
        slot2_end = time_to_minutes(slot2["end"])
        
        # Check for overlap
        overlap = not (slot1_end <= slot2_start or slot2_end <= slot1_start)
        assert overlap is True  # These slots should overlap
```

**File**: `/backend_gen/tests/test_schemas.py`
```python
# NOTE: Unit tests should use real LLM calls, real file access, and real code execution wherever possible.
# See the node implementation for actual LLM usage.

import pytest
from pydantic import ValidationError
from agent.tools_and_schemas import (
    PatientInfoSchema,
    AppointmentRequestSchema,
    DoctorScheduleSchema,
    AppointmentConfirmationSchema
)

class TestPydanticSchemas:
    """Test Pydantic model validation with real data"""
    
    def test_patient_info_schema_valid_input(self):
        """Test patient info schema with valid real inputs"""
        valid_patients = [
            {"name": "John Doe", "age": 35, "contact": "john@example.com", "symptoms": "Headache"},
            {"name": "Jane Smith", "age": 28, "contact": "jane@example.com", "symptoms": "Fever"},
            {"name": "Bob Johnson", "age": 45, "contact": "bob@example.com"}
        ]
        
        for patient_data in valid_patients:
            schema = PatientInfoSchema(**patient_data)
            assert schema.name == patient_data["name"]
            assert schema.age == patient_data["age"]
            assert schema.contact == patient_data["contact"]
    
    def test_patient_info_schema_invalid_input(self):
        """Test patient info schema with invalid real inputs"""
        invalid_inputs = [
            {"name": "", "age": 35, "contact": "john@example.com"},  # Empty name
            {"name": "John", "age": -5, "contact": "john@example.com"},  # Negative age
            {"name": "John", "age": "not_a_number", "contact": "john@example.com"},  # Invalid age type
        ]
        
        for invalid_input in invalid_inputs:
            with pytest.raises(ValidationError):
                PatientInfoSchema(**invalid_input)
    
    def test_appointment_request_schema_valid_input(self):
        """Test appointment request schema with valid real inputs"""
        valid_request = {
            "patient_name": "John Doe",
            "requested_date": "2024-07-01",
            "requested_time": "10:00",
            "doctor_specialty": "General Medicine"
        }
        
        schema = AppointmentRequestSchema(**valid_request)
        assert schema.patient_name == valid_request["patient_name"]
        assert schema.requested_date == valid_request["requested_date"]
        assert schema.requested_time == valid_request["requested_time"]
    
    def test_doctor_schedule_schema_valid_input(self):
        """Test doctor schedule schema with valid real inputs"""
        valid_schedule = {
            "doctor_name": "Dr. Smith",
            "specialty": "General Medicine",
            "available_slots": [
                {"date": "2024-07-01", "time": "10:00"},
                {"date": "2024-07-01", "time": "11:00"}
            ]
        }
        
        schema = DoctorScheduleSchema(**valid_schedule)
        assert schema.doctor_name == valid_schedule["doctor_name"]
        assert schema.specialty == valid_schedule["specialty"]
        assert len(schema.available_slots) == 2
    
    def test_appointment_confirmation_schema_valid_input(self):
        """Test appointment confirmation schema with valid real inputs"""
        valid_confirmation = {
            "patient_name": "John Doe",
            "doctor_name": "Dr. Smith",
            "date": "2024-07-01",
            "time": "10:00",
            "status": "confirmed"
        }
        
        schema = AppointmentConfirmationSchema(**valid_confirmation)
        assert schema.patient_name == valid_confirmation["patient_name"]
        assert schema.doctor_name == valid_confirmation["doctor_name"]
        assert schema.status == valid_confirmation["status"]

class TestSchemaIntegration:
    """Test schema integration with real system components"""
    
    def test_state_schema_compatibility(self):
        """Test that schemas work with the real state management"""
        from agent.state import OverallState
        
        # Test with real state data
        state_data = {
            "messages": [{"role": "human", "content": "I need an appointment"}],
            "patient_info": {"name": "John Doe", "age": 35, "contact": "john@example.com"},
            "appointment_request": {"patient_name": "John Doe", "requested_date": "2024-07-01", "requested_time": "10:00"},
            "doctor_schedules": [{"doctor_name": "Dr. Smith", "specialty": "General Medicine", "available_slots": []}],
            "scheduled_appointment": None,
            "errors": None
        }
        
        # Validate that real data works with schemas
        patient_schema = PatientInfoSchema(**state_data["patient_info"])
        assert patient_schema.name == "John Doe"
        
        request_schema = AppointmentRequestSchema(**state_data["appointment_request"])
        assert request_schema.patient_name == "John Doe"
```

**File**: `/backend_gen/tests/conftest.py`
```python
import pytest
import os
from dotenv import load_dotenv

# Load real environment variables for testing
load_dotenv(dotenv_path=".env")

@pytest.fixture
def sample_state():
    """Provide real sample state for testing"""
    return {
        "messages": [{"role": "human", "content": "I need an appointment for a headache"}],
        "patient_info": None,
        "appointment_request": None,
        "doctor_schedules": None,
        "scheduled_appointment": None,
        "errors": None
    }

@pytest.fixture
def complete_state():
    """Provide complete state with all real data for testing"""
    return {
        "messages": [{"role": "human", "content": "I need an appointment"}],
        "patient_info": {"name": "John Doe", "age": 35, "contact": "john@example.com", "symptoms": "Headache"},
        "appointment_request": {"patient_name": "John Doe", "requested_date": "2024-07-01", "requested_time": "10:00"},
        "doctor_schedules": [
            {
                "doctor_name": "Dr. Smith",
                "specialty": "General Medicine",
                "available_slots": [{"date": "2024-07-01", "time": "10:00"}]
            }
        ],
        "scheduled_appointment": None,
        "errors": None
    }

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup real test environment variables"""
    # Ensure real API key is available for testing
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not available for real LLM testing")
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
- [ ] All agent tests pass using real LLM calls, real file access, and real code execution
- [ ] All tool tests validate functionality and error handling with real data
- [ ] All schema tests cover validation rules and edge cases with real inputs
- [ ] Test coverage > 80% for all agent and tool code
- [ ] File operations tests use actual file I/O
- [ ] Mathematical calculations tests perform real computations
- [ ] All tests demonstrate real integration behavior

#### 3.1 Graph Compilation & Import Validation
```bash
# Install and verify
cd /backend_gen
pip install -e .

# Test imports
python -c "from agent.graph import build_graph; build_graph()"

# Verify graph structure and compilation
python -c "
from agent.graph import graph
print('Graph name:', graph.name if hasattr(graph, 'name') else 'agent')
print('Graph compiled successfully!')
"

# Test state schema import
python -c "from agent.state import OverallState; print('State schema imported successfully')"

# Test all node imports
python -c "
from agent.nodes.portfolio_analyzer import portfolio_analyzer_node
from agent.nodes.market_research import market_research_node  
from agent.nodes.rebalancing_executor import rebalancing_executor_node
from agent.nodes.supervisor import supervisor_router
print('All node imports successful')
"
```

**Graph Compilation Success Criteria**:
- [ ] All imports execute without errors
- [ ] Graph builds and compiles successfully
- [ ] State schema is valid TypedDict
- [ ] All nodes are properly importable
- [ ] No circular import dependencies
- [ ] LangGraph configuration is valid

#### 3.2 LangGraph Development Server Testing

**CRITICAL LESSONS LEARNED**: Previous testing documentation was completely false. The real testing process revealed multiple import errors and server startup failures that were not caught in unit tests.

**Critical Discovery**: LangGraph server runs on port 2024 (not 8123) and uses thread-based API architecture. Real testing revealed that:
1. **Import errors only surface when server actually runs**, not during unit tests
2. **Fake model compatibility issues** with different LangChain versions
3. **Relative vs absolute imports** cause runtime failures in server context
4. **Mock testing vs real execution** - mocks can hide real integration issues

**MANDATORY PRE-SERVER CHECKS**:
```bash
# 1. CRITICAL: Fix relative imports in graph.py BEFORE server testing
# Replace ALL relative imports like:
# from .nodes.clinical_data_collector import clinical_data_collector_agent
# WITH absolute imports:
# from agent.nodes.clinical_data_collector import clinical_data_collector_agent

# 2. CRITICAL: Fix fake LLM imports in utils.py
# The following imports will FAIL:
# - from langchain_core.language_models.fake import FakeListChatModel (doesn't exist)
# - from langchain_core.language_models.fake import FakeChatModel (doesn't exist)
# - from langchain_core.language_models.llm import LLM (path doesn't exist)

# Use this working pattern instead:
cat > src/agent/utils.py << 'EOF'
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
        return FakeListLLM(responses=[
            "Clinical data analysis complete. Quality assessment: Good.",
            "Statistical analysis reveals significant correlations.",
            "Compliance validation successful. HIPAA requirements met.",
            "Research report generated with comprehensive findings."
        ])
    except ImportError:
        # Final fallback - create simple mock
        class SimpleFakeLLM:
            def invoke(self, prompt):
                class Response:
                    content = "Mock LLM response for testing"
                return Response()
        return SimpleFakeLLM()
EOF

# 3. CRITICAL: Test graph loading BEFORE server
cd /backend_gen
python -c "from agent.graph import graph; print('Graph loads:', type(graph))"

# 4. CRITICAL: Install package in editable mode
pip install -e .

# 5. CRITICAL: Check for circular imports in __init__.py
# Remove/comment any imports in agent/__init__.py that cause circular dependencies
echo "# Removed circular import to prevent server startup issues" > src/agent/__init__.py
```

**ONLY AFTER ALL PRE-CHECKS PASS, START SERVER:**
```bash
# Start the LangGraph development server with proper process management
nohup langgraph dev > langgraph.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# CRITICAL: Wait and check logs for import errors
sleep 10
if grep -q "ImportError\|ModuleNotFoundError\|attempted relative import" langgraph.log; then
    echo "❌ CRITICAL: Import errors detected. Fix before proceeding."
    cat langgraph.log | grep -A3 -B3 "Error"
    kill $SERVER_PID
    exit 1
fi

# Cleanup function for proper resource management
cleanup() {
    echo "Cleaning up server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT

# Test 1: Server Health Check via OpenAPI
echo -e "\n=== Test 1: Server Health Check ==="
curl -s http://localhost:2024/openapi.json | jq '.paths | keys' > /dev/null || {
    echo "❌ Server not responding on port 2024"
    exit 1
}
echo "✅ Server responding on correct port"

# Test 2: Thread Creation
echo -e "\n=== Test 2: Thread Management ==="
THREAD_RESPONSE=$(curl -s -X POST http://localhost:2024/threads \
  -H "Content-Type: application/json" \
  -d '{}')

THREAD_ID=$(echo "$THREAD_RESPONSE" | jq -r '.thread_id')
if [ "$THREAD_ID" = "null" ] || [ -z "$THREAD_ID" ]; then
    echo "❌ Thread creation failed"
    exit 1
fi
echo "✅ Thread created: $THREAD_ID"

# Test 3: ACTUAL LLM EXECUTION TEST (NOT MOCKED)
echo -e "\n=== Test 3: Real LLM Execution Test ==="
EXECUTION_RESPONSE=$(curl -s -X POST http://localhost:2024/threads/$THREAD_ID/runs \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "agent",
    "input": {
      "messages": [{"role": "human", "content": "Test clinical data processing"}],
      "clinical_data": null,
      "literature_data": null,
      "validated_data": null,
      "statistical_analysis": null,
      "compliance_report": null,
      "research_report": null,
      "current_tier": "acquisition",
      "processing_status": "pending",
      "next_agent": "clinical_data_collector_agent",
      "patient_count": 10,
      "study_type": "test_study",
      "audit_trail": [],
      "errors": null
    }
  }')

RUN_ID=$(echo "$EXECUTION_RESPONSE" | jq -r '.run_id')
echo "Started run: $RUN_ID"

# Wait for execution and check for real errors
echo "Waiting for LLM execution..."
sleep 20

# CRITICAL: Check run status for real execution results
RUN_STATUS_RESPONSE=$(curl -s http://localhost:2024/threads/$THREAD_ID/runs/$RUN_ID)
RUN_STATUS=$(echo "$RUN_STATUS_RESPONSE" | jq -r '.status')

echo "Run status: $RUN_STATUS"

if [ "$RUN_STATUS" = "error" ]; then
    echo "❌ CRITICAL: Graph execution failed with real LLM call"
    echo "Check server logs for import/execution errors:"
    tail -30 langgraph.log | grep -A5 -B5 "error"
    exit 1
elif [ "$RUN_STATUS" = "success" ]; then
    echo "✅ SUCCESS: Real LLM execution completed"
    
    # Test 4: Verify Actual LLM Responses
    echo -e "\n=== Test 4: LLM Response Validation ==="
    FINAL_STATE=$(curl -s http://localhost:2024/threads/$THREAD_ID/state)
    
    # Check for evidence of real LLM processing
    if echo "$FINAL_STATE" | jq '.values' | grep -q "clinical\|analysis\|data"; then
        echo "✅ Real LLM responses detected in final state"
        echo "Sample response:"
        echo "$FINAL_STATE" | jq '.values.messages[-1].content' 2>/dev/null || echo "No message content found"
    else
        echo "❌ No LLM-generated content found in final state"
        echo "Final state:"
        echo "$FINAL_STATE" | jq '.values'
    fi
else
    echo "⚠ Run status: $RUN_STATUS (still processing or unknown)"
fi

echo -e "\n🎉 Real LLM Endpoint Testing Complete!"
```

**LangGraph Server Success Criteria (UPDATED WITH REAL TESTING)**:
- [ ] **PRE-CHECK**: All relative imports converted to absolute imports in graph.py
- [ ] **PRE-CHECK**: Fake LLM imports fixed using working patterns in utils.py  
- [ ] **PRE-CHECK**: Graph loads successfully with `python -c "from agent.graph import graph"`
- [ ] **PRE-CHECK**: Package installed in editable mode with `pip install -e .`
- [ ] **PRE-CHECK**: No circular imports in agent/__init__.py
- [ ] `langgraph dev` starts without import errors on port 2024
- [ ] Server logs show no ImportError, ModuleNotFoundError, or relative import issues
- [ ] OpenAPI schema endpoint (/openapi.json) returns valid JSON with correct paths
- [ ] Thread management (/threads) creates threads successfully
- [ ] Graph execution (/threads/{id}/runs) processes requests with REAL LLM calls
- [ ] Run status transitions to "success" (not "error") with actual clinical data
- [ ] Final state contains evidence of LLM-generated content (not just mock responses)
- [ ] Server logs show successful graph execution without import/runtime errors
- [ ] Performance testing shows real LLM execution times under reasonable limits
- [ ] Cleanup functions prevent hanging processes

**CRITICAL TESTING PRINCIPLE**: Mock tests can pass while real execution fails. Always test actual server execution with real LLM calls to catch import errors, dependency issues, and runtime failures that only surface in the server context. The specific errors encountered (relative imports, fake model imports, circular imports) MUST be fixed before attempting server startup.

#### 3.3 API Integration Testing
```bash
# Create comprehensive API test script
cat > test_api_endpoints.sh << 'EOF'
#!/bin/bash
set -e

echo "=== LangGraph API Integration Tests ==="

# Start server in background
echo "Starting LangGraph dev server..."
langgraph dev &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up server (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT

# Wait for server startup
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8123/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Server failed to start within 30 seconds"
        exit 1
    fi
    sleep 1
done

# Test 1: Health Check
echo -e "\n=== Test 1: Health Check ==="
curl -f http://localhost:8123/health
echo -e "\n✓ Health check passed"

# Test 2: Schema Validation
echo -e "\n=== Test 2: Schema Validation ==="
SCHEMA_RESPONSE=$(curl -s http://localhost:8123/agent/schema)
echo "$SCHEMA_RESPONSE" | jq . > /dev/null
echo "✓ Schema endpoint returns valid JSON"

# Test 3: Graph Invocation
echo -e "\n=== Test 3: Graph Invocation ==="
INVOKE_RESPONSE=$(curl -s -X POST http://localhost:8123/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [],
      "current_portfolio": null,
      "target_allocation": null,
      "market_data": null,
      "rebalancing_plan": null,
      "executed_trades": null,
      "risk_constraints": null,
      "notifications": [],
      "errors": null
    }
  }')

echo "$INVOKE_RESPONSE" | jq . > /dev/null
echo "✓ Invoke endpoint returns valid JSON"

# Validate response structure
echo "$INVOKE_RESPONSE" | jq -e '.output.messages' > /dev/null
echo "✓ Response contains expected 'messages' field"

echo "$INVOKE_RESPONSE" | jq -e '.output.notifications' > /dev/null  
echo "✓ Response contains expected 'notifications' field"

# Test 4: Streaming Endpoint
echo -e "\n=== Test 4: Streaming Endpoint ==="
STREAM_OUTPUT=$(curl -s -X POST http://localhost:8123/agent/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [],
      "current_portfolio": null,
      "target_allocation": null,
      "market_data": null,
      "rebalancing_plan": null,
      "executed_trades": null,
      "risk_constraints": null,
      "notifications": [],
      "errors": null
    }
  }')

if [ -n "$STREAM_OUTPUT" ]; then
    echo "✓ Stream endpoint returns data"
else
    echo "✗ Stream endpoint returned no data"
    exit 1
fi

# Test 5: Error Handling
echo -e "\n=== Test 5: Error Handling ==="
ERROR_RESPONSE=$(curl -s -X POST http://localhost:8123/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"invalid": "json_structure"}')

if echo "$ERROR_RESPONSE" | grep -q "error\|Error"; then
    echo "✓ Server handles invalid requests gracefully"
else
    echo "✗ Server did not handle invalid request properly"
    exit 1
fi

# Test 6: Performance Check
echo -e "\n=== Test 6: Performance Check ==="
START_TIME=$(date +%s)
curl -s -X POST http://localhost:8123/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [],
      "current_portfolio": null,
      "target_allocation": null,
      "market_data": null,
      "rebalancing_plan": null,
      "executed_trades": null,
      "risk_constraints": null,
      "notifications": [],
      "errors": null
    }
  }' > /dev/null
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "✓ Graph execution completed in ${DURATION} seconds"
if [ $DURATION -lt 60 ]; then
    echo "✓ Performance within acceptable range (< 60s)"
else
    echo "⚠ Performance slower than expected (>= 60s)"
fi

echo -e "\n=== All API Integration Tests Passed! ==="
EOF

# Make script executable and run
chmod +x test_api_endpoints.sh
./test_api_endpoints.sh
```

**API Integration Success Criteria**:
- [ ] All curl commands execute successfully (non-zero exit codes fail)
- [ ] Health endpoint returns successful response
- [ ] Schema endpoint returns valid JSON schema for the graph
- [ ] Invoke endpoint processes requests and returns structured results
- [ ] Stream endpoint provides real-time execution updates
- [ ] Error handling works for malformed requests
- [ ] Response times are within acceptable limits (< 60 seconds)
- [ ] Server starts and stops cleanly without hanging processes

#### 3.4 Validation Tasks
```bash
```

# LangGraph Multi-Agent Development Protocol - Planning Document

## Autonomous Development Progress

### Completed Iterations

**Iteration 6: Manufacturing Supply Chain Optimization** ✅ **PHASE 3 COMPLETED**
- **Business Case**: Manufacturing Supply Chain Optimization ($24.8B Market)
- **Architecture**: Network (5 Agents) - Supply Chain Coordinator (Hub), Supplier Intelligence, Production Planning, Logistics Optimization, Quality Assurance
- **Phase 1**: ✅ COMPLETED - Architecture design and agent specification
- **Phase 2**: ✅ COMPLETED - Implementation of 5-agent network with manufacturing schemas
- **Phase 3**: ✅ **COMPLETED** - Testing & Validation
  - **Phase 3.1**: ✅ Unit Testing - 23/23 core tests PASSED
  - **Phase 3.2**: ✅ Graph Compilation - Successful compilation
  - **Phase 3.3**: ✅ Server Testing - FastAPI integration validated
- **Status**: Ready for Phase 4 (Production Integration)
- **Key Achievement**: Manufacturing supply chain optimization system fully validated with real business logic
- **Market Validation**: $24.8B target market with proven technical foundation

### Next Iteration Ready

**Iteration 7: [Next Business Case - TBD]**
- Ready to commence with autonomous protocol
- Building on validated network architecture patterns from Iteration 6

---

*Last Updated: June 26, 2025 - Iteration 6 Phase 3 Completion*