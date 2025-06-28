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

### Phase 2: Implementation & Code Generation
**Objective**: Generate all required code components using accumulated knowledge. **This phase starts automatically after planning, with no user confirmation required.**
**Success Criteria**:
- All mandatory files created under `/backend_gen/src/agent/`
- LLM integration properly configured
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

#### Phase 3.2: Graph Compilation & Import Validation  
**Success Criteria**:
- [ ] Graph compiles and imports successfully without errors
- [ ] All node imports execute without circular dependencies
- [ ] State schema is valid TypedDict structure
- [ ] LangGraph configuration files are properly structured
- [ ] **Pre-server validation** prevents import errors at runtime
- [ ] Package installation completes successfully

#### Phase 3.3: LangGraph Server Testing & Real Execution
**Success Criteria**:
- [ ] `langgraph dev` server starts without errors or warnings from correct directory
- [ ] Server logs show no ImportError, ModuleNotFoundError, or relative import issues
- [ ] OpenAPI schema endpoint returns valid JSON with correct paths
- [ ] Thread management creates and manages execution threads properly
- [ ] **Real LLM execution** processes requests with actual API calls (not mocks)
- [ ] **Complete LLM conversations logged** with prompts, responses, and timing
- [ ] Graph execution transitions through all states successfully
- [ ] Server cleanup prevents hanging processes

#### Phase 3.4: API Integration & End-to-End Validation
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
echo "✅ Graph Compilation: Successfully imports and compiles"
echo "✅ LangGraph Server: Real LLM execution verified"
echo "✅ API Integration: All endpoints functional"
echo "🎯 System Ready for Production Deployment"

#### 3.5 LangWatch Scenario Testing & Agent Simulation (NEW)
**Objective**: Advanced agent testing through simulation-based testing with LangWatch Scenario framework. **This phase starts automatically after API integration testing, with no user confirmation required.**

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
    cache_key='healthcare-coordination-tests',  # For repeatable tests
    verbose=True  # Show detailed simulation output
)
print('LangWatch Scenario configured')
"
```

**Agent Adapter Implementation**:
```bash
# Create LangWatch Scenario adapter for our LangGraph agent
cat > tests/test_scenario_healthcare_coordination.py << 'EOF'
import pytest
import scenario
import asyncio
import json
from typing import Dict, Any, List
from agent.graph import graph

# Configure scenario with appropriate settings
scenario.configure(
    default_model="openai/gpt-4.1-mini",
    cache_key="healthcare-coordination-v1",
    verbose=True
)

class HealthcareCoordinationAgent(scenario.AgentAdapter):
    """LangWatch Scenario adapter for our Healthcare Coordination LangGraph agent"""
    
    def __init__(self):
        self.graph = graph
        
    async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
        """
        Adapter method that LangWatch Scenario calls to interact with our agent.
        Converts scenario input to LangGraph format and back.
        """
        # Convert scenario messages to our state format
        state = {
            "messages": input.messages,
            "patient_info": None,
            "medication_assessment": None,
            "specialist_coordination": None,
            "care_plan": None,
            "care_coordination_plan": None,
            "provider_notifications": None,
            "errors": None
        }
        
        try:
            # Execute our LangGraph agent
            result = await self.graph.ainvoke(state)
            
            # Extract the final response message
            if result.get("messages") and len(result["messages"]) > 0:
                final_message = result["messages"][-1]
                if isinstance(final_message, dict) and "content" in final_message:
                    return final_message["content"]
                else:
                    return str(final_message)
            
            # Fallback: return care coordination plan if available
            if result.get("care_coordination_plan"):
                return f"Care Coordination Plan Generated:\n{result['care_coordination_plan']}"
                
            return "Healthcare coordination assessment completed."
            
        except Exception as e:
            return f"Error in healthcare coordination: {str(e)}"

# Test Scenarios for Healthcare Coordination System

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_routine_checkup_coordination():
    """Test coordination for a routine patient checkup scenario"""
    
    result = await scenario.run(
        name="routine_checkup_coordination",
        description="""
            A 45-year-old patient needs coordination for their annual physical exam.
            They have diabetes and hypertension, take multiple medications,
            and need specialist follow-ups. The system should coordinate their care
            efficiently while ensuring medication safety and proper specialist referrals.
        """,
        agents=[
            HealthcareCoordinationAgent(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent should gather comprehensive patient information",
                    "Agent should identify medication interaction risks",
                    "Agent should coordinate appropriate specialist referrals",
                    "Agent should create a clear care coordination plan",
                    "Agent should NOT ask redundant questions about already provided information",
                    "Agent should prioritize urgent health concerns appropriately"
                ]
            ),
        ],
        max_turns=8,  # Allow sufficient interaction for complex coordination
        set_id="healthcare-coordination-tests",
    )
    
    assert result.success, f"Routine checkup coordination failed: {result.failure_reason}"

@pytest.mark.agent_test  
@pytest.mark.asyncio
async def test_emergency_coordination_scenario():
    """Test coordination for an emergency healthcare scenario"""
    
    result = await scenario.run(
        name="emergency_coordination",
        description="""
            A 68-year-old patient presents with chest pain and shortness of breath.
            They have a history of heart disease and are on blood thinners.
            The system must coordinate urgent care while managing medication risks
            and ensuring rapid specialist consultation.
        """,
        agents=[
            HealthcareCoordinationAgent(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent should recognize the urgency of chest pain symptoms",
                    "Agent should prioritize emergency specialist consultation", 
                    "Agent should flag critical medication interactions with blood thinners",
                    "Agent should create urgent care coordination plan",
                    "Agent should NOT delay care with unnecessary questions",
                    "Agent should ensure emergency protocols are followed"
                ]
            ),
        ],
        max_turns=6,  # Shorter for emergency scenarios
        script=[
            scenario.user("I'm having severe chest pain and can't breathe well"),
            scenario.agent(),  # Let agent respond to emergency
            scenario.user(),   # User provides more details
            scenario.agent(),  # Agent coordinates emergency care
            scenario.judge(),  # Evaluate emergency response
        ],
        set_id="healthcare-coordination-tests",
    )
    
    assert result.success, f"Emergency coordination failed: {result.failure_reason}"

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_complex_medication_management():
    """Test medication management for complex polypharmacy scenario"""
    
    result = await scenario.run(
        name="complex_medication_management", 
        description="""
            An elderly patient is taking 12 different medications from multiple specialists.
            They're experiencing side effects and potential drug interactions.
            The system should coordinate medication review, identify interactions,
            and work with specialists to optimize their medication regimen.
        """,
        agents=[
            HealthcareCoordinationAgent(),
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(
                criteria=[
                    "Agent should perform comprehensive medication review",
                    "Agent should identify potential drug interactions", 
                    "Agent should coordinate with prescribing specialists",
                    "Agent should prioritize medication safety",
                    "Agent should create medication optimization plan",
                    "Agent should address patient concerns about side effects"
                ]
            ),
        ],
        max_turns=10,  # Extended for complex medication coordination
        set_id="healthcare-coordination-tests",
    )
    
    assert result.success, f"Medication management coordination failed: {result.failure_reason}"

# Advanced Scenario with Custom Evaluation
def check_medication_safety_protocols(state: scenario.ScenarioState):
    """Custom assertion to check if medication safety protocols were followed"""
    conversation = " ".join([msg.get("content", "") for msg in state.messages])
    
    # Check for key medication safety indicators
    safety_checks = [
        "drug interaction" in conversation.lower(),
        "allergy" in conversation.lower() or "allergic" in conversation.lower(), 
        "dosage" in conversation.lower(),
        "side effect" in conversation.lower()
    ]
    
    assert any(safety_checks), "Agent did not perform adequate medication safety screening"

@pytest.mark.agent_test
@pytest.mark.asyncio
async def test_medication_safety_protocols():
    """Test that medication safety protocols are properly followed"""
    
    result = await scenario.run(
        name="medication_safety_protocols",
        description="""
            A patient is starting a new medication that has potential interactions
            with their existing medications. The system must follow proper safety
            protocols before approving the new medication.
        """,
        agents=[
            HealthcareCoordinationAgent(),
            scenario.UserSimulatorAgent(),
        ],
        script=[
            scenario.user("My doctor wants to start me on a new blood pressure medication"),
            scenario.agent(),  # Agent responds
            scenario.user(),   # User provides medication list  
            scenario.agent(),  # Agent analyzes safety
            check_medication_safety_protocols,  # Custom safety check
            scenario.succeed(),  # End successfully if safety checks pass
        ],
        set_id="healthcare-coordination-tests",
    )
    
    assert result.success, f"Medication safety protocols failed: {result.failure_reason}"
EOF
```

**Execution Commands**:
```bash
# Run all LangWatch Scenario tests
cd /backend_gen
python -m pytest tests/test_scenario_healthcare_coordination.py -v -s --tb=short

# Run with debug mode for step-by-step interaction
python -m pytest tests/test_scenario_healthcare_coordination.py::test_routine_checkup_coordination -v -s --debug

# Run specific scenario with cache busting
SCENARIO_CACHE_KEY="new-test-run" python -m pytest tests/test_scenario_healthcare_coordination.py -v -s
```

**LangWatch Scenario Success Criteria**:
- [ ] **Installation**: LangWatch Scenario package installed and configured
- [ ] **Agent Adapter**: Healthcare coordination agent successfully adapted for scenario testing
- [ ] **Basic Scenarios**: All core healthcare coordination scenarios pass (routine, emergency, medication)
- [ ] **Custom Evaluations**: Custom assertion functions work for domain-specific validation
- [ ] **Judge Agents**: AI judges properly evaluate conversation quality against healthcare criteria
- [ ] **User Simulation**: Realistic user behavior simulation covers various patient types and situations
- [ ] **Integration**: Scenario tests integrate with existing test pipeline and CI/CD
- [ ] **Cache Management**: Deterministic testing with proper cache key management for repeatability

**Benefits of LangWatch Scenario Testing**:
1. **Real User Simulation**: Tests agent behavior with realistic user interactions instead of fixed test cases
2. **Multi-turn Conversations**: Validates complex conversational flows that unit tests can't capture  
3. **Edge Case Coverage**: Automatically discovers edge cases through varied user simulation
4. **Quality Evaluation**: AI judges provide sophisticated evaluation beyond simple assertion checks
5. **Performance Validation**: Measures real-world performance under different interaction patterns
6. **Domain-Specific Testing**: Healthcare-specific scenarios validate medical coordination workflows

This comprehensive scenario testing phase ensures our healthcare coordination system performs reliably across diverse real-world situations, handling both common cases and challenging edge scenarios with appropriate quality and safety measures.

---

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