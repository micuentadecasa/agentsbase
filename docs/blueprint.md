Of course. This level of explicit detail is paramount for creating a truly automated and reliable development system. Each phase will be treated as a discrete, self-contained task with its own inputs, outputs, and rigorous validation checks.

Here is the master blueprint, fully expanded with concrete examples for all artifacts, generated code, documentation, and a comprehensive suite of validation checks suitable for an automated tooling environment.

---

# The Declarative Agent Development & Operations Blueprint (v4.0 - Automated)

This document outlines a rigorous, artifact-driven process for building and maintaining LangGraph agents. It is designed for an automated tooling environment capable of file I/O, code generation, linting, and test execution.

**Running Example:** The **Pro Search Agent**.

---

## Part 1: Full Agent Creation Lifecycle (From Scratch)

### Phase 0: Workspace Preparation & Bootstrapping

**Objective:** To establish a clean, consistent development workspace from a baseline.

#### Task Summary:
1.  **Check:** Ensure `/backend/` exists and is not empty.
2.  **Clean:** Delete the `/backend_gen/` directory if it exists.
3.  **Bootstrap:** Copy the entire contents of `/backend/` into a new `/backend_gen/` directory.

#### Example Output Documentation (`/docs/instructions/00_workspace_preparation.md`):
```md
# Workspace Preparation & Bootstrapping

This document confirms that the initial workspace setup for the agent has been completed.

### Actions Performed:

1.  The `/backend_gen/` directory has been purged of any previous content.
2.  A fresh copy of the baseline application code from `/backend/` has been placed into `/backend_gen/`.

The system is now ready for the agent specification and code generation phases. All subsequent modifications will occur within the `/backend_gen/` directory.
```

#### ✅ Expanded Validation Checklist (for automated tool):
1.  **Path Existence Check:**
    *   `assert os.path.isdir("./backend")` -> PASS/FAIL
2.  **Clean Operation:**
    *   Run `shutil.rmtree("./backend_gen", ignore_errors=True)`.
    *   `assert not os.path.isdir("./backend_gen")` -> PASS/FAIL
3.  **Copy Operation:**
    *   Run `shutil.copytree("./backend", "./backend_gen")`.
    *   Run a file hash comparison (`hashlib`) on a key file (e.g., `/backend/app.py` vs `/backend_gen/app.py`) to confirm a successful copy. -> PASS/FAIL

---

### Phase 1: Agent Specification

**Objective:** To translate the high-level use case into a formal graph definition YAML.

#### Task Summary:
1.  **Input:** Read the high-level requirement document located at `/docs/agent_use_case.md`.
2.  **Output:** Create the master workflow file at `/tasks/artifacts/agent_spec.yaml`.
3.  **Output:** Create the validation instructions at `/docs/instructions/01_validate_agent_spec.md`.

#### Example Output Artifact (`/tasks/artifacts/agent_spec.yaml`):
```yaml
# This file is the single source of truth for the agent's control flow.
agent_name: ProSearchAgent
goal: "Answer a user's question via iterative web research and synthesis."

entry_point: generate_query

nodes:
  - name: generate_query
    description: "Generates initial search queries based on the user's question."
  - name: web_research
    description: "Performs a web search for a single query and synthesizes results."
  - name: reflection
    description: "Analyzes all gathered research to find knowledge gaps."
  - name: finalize_answer
    description: "Compiles all research into a final, cited answer."

edges:
  - from: web_research
    to: reflection
  - from: finalize_answer
    to: END

conditional_edges:
  - from: generate_query
    condition_func: continue_to_web_research
    paths: [web_research]
  - from: reflection
    condition_func: evaluate_research
    paths: [web_research, finalize_answer]
```

#### Example Output Documentation (`/docs/instructions/01_validate_agent_spec.md`):
```md
# How to Validate the Agent Specification

The `agent_spec.yaml` file defines the agent's core structure. To ensure it is logically sound before proceeding, run the validation script.

### Steps:

1.  Navigate to the project's root directory.
2.  Run the validation command:
    ```bash
    python backend_gen/validation/run_validations.py --check-spec
    ```
3.  **Expected Outcome:** The script should complete with exit code 0 and print "Agent spec validation PASSED."
4.  **Manual Review:**
    *   Does the `entry_point` node exist?
    *   Do all `from` and `to` fields in `edges` and `paths` refer to declared nodes or `END`?
```

#### ✅ Expanded Validation Checklist:
1.  **File Existence:** `assert os.path.isfile("./tasks/artifacts/agent_spec.yaml")` -> PASS/FAIL
2.  **YAML Parsing:** Attempt to load the file with `pyyaml.safe_load()`. A `YAMLError` is a FAIL.
3.  **Schema & Integrity Check:**
    *   Check for required top-level keys: `agent_name`, `entry_point`, `nodes`.
    *   Create a set of all `node['name']` values.
    *   Verify `entry_point` is in the node set.
    *   Iterate through all `edges` and `conditional_edges`: verify that every `from` and `to`/`path` value is in the node set or is `END`. -> FAIL if any node is referenced but not declared.
4.  **Graph Connectivity (Orphan Check):**
    *   Perform a graph traversal (like BFS or DFS) starting from `entry_point`.
    *   Collect all reachable nodes.
    *   Assert that the set of reachable nodes is equal to the set of all declared nodes. -> FAIL if an "orphaned" node exists.

---

### Phase 2: State Specification

**Objective:** To formally define the agent's memory and auto-generate the corresponding Python code.

#### Task Summary:
1.  **Input:** Read `/tasks/artifacts/agent_spec.yaml`.
2.  **Output:** Create `/tasks/artifacts/state_spec.yaml`.
3.  **Output:** Generate `/backend_gen/agent/state.py`.
4.  **Output:** Create `/docs/instructions/02_validate_state_spec.md`.

#### Example Output Artifact (`/tasks/artifacts/state_spec.yaml`):
```yaml
state_variables:
  - name: messages
    type: list
    description: "The history of the conversation, managed by LangGraph."
    accumulator: add_messages
  - name: search_query
    type: list
    description: "A list of all web search queries that have been run."
    accumulator: operator.add
  - name: web_research_result
    type: list
    description: "A list of synthesized text from each web search."
    accumulator: operator.add
  - name: sources_gathered
    type: list
    description: "A list of citation source objects."
    accumulator: operator.add
  - name: research_loop_count
    type: int
    description: "Counter for the research loop to prevent infinite cycles."
    accumulator: overwrite
```

#### Example Generated Code (`/backend_gen/agent/state.py`):
```python
# This file is auto-generated from /tasks/artifacts/state_spec.yaml.
# Do not edit this file directly.

from typing import TypedDict, List
from typing_extensions import Annotated
import operator
from langgraph.graph import add_messages

class OverallState(TypedDict):
    """The complete state of the ProSearchAgent."""

    # The history of the conversation, managed by LangGraph.
    messages: Annotated[List, add_messages]

    # A list of all web search queries that have been run.
    search_query: Annotated[List, operator.add]

    # A list of synthesized text from each web search.
    web_research_result: Annotated[List, operator.add]

    # A list of citation source objects.
    sources_gathered: Annotated[List, operator.add]

    # Counter for the research loop to prevent infinite cycles.
    research_loop_count: int
```

#### Example Output Documentation (`/docs/instructions/02_validate_state_spec.md`):
```md
# How to Validate and Generate the Agent State

The `state_spec.yaml` file is the source of truth for the agent's memory. This process validates the spec and generates the Python code from it.

### Steps:
1.  Run the generation command:
    ```bash
    python backend_gen/validation/run_validations.py --generate-state
    ```
2.  **Expected Outcome:** The script should succeed. A file will be created/updated at `/backend_gen/agent/state.py`.
3.  **Manual Review:** Open `/backend_gen/agent/state.py`. Does the `OverallState` `TypedDict` match the spec?
```

#### ✅ Expanded Validation Checklist:
1.  **File Existence:** `assert os.path.isfile("./tasks/artifacts/state_spec.yaml")` -> PASS/FAIL
2.  **YAML Parsing & Schema Check:** Load YAML and verify each item in `state_variables` has `name`, `type`, and `accumulator`.
3.  **Code Generation Check:** After running the generator, `assert os.path.isfile("./backend_gen/agent/state.py")` -> PASS/FAIL
4.  **Linting:** Run `black ./backend_gen/agent/state.py --check` and `ruff check ./backend_gen/agent/state.py`. -> FAIL on any linting errors.
5.  **Type Checking:** Run `mypy ./backend_gen/agent/state.py`. -> FAIL on any type errors.

---

### Phase 3: Tools & Schema Specification

**Objective:** To define the interfaces for LLM structured outputs and external tools.

#### Task Summary:
1.  **Input:** Read `/tasks/artifacts/agent_spec.yaml`.
2.  **Output:** Create `/tasks/artifacts/tools_spec.yaml`.
3.  **Output:** Generate `/backend_gen/agent/tools_and_schemas.py`.
4.  **Output:** Create `/docs/instructions/03_validate_tools_spec.md`.

#### Example Output Artifact (`/tasks/artifacts/tools_spec.yaml`):
```yaml
schemas:
  - name: SearchQueryList
    description: "A structured list of web search queries and the rationale."
    fields:
      - name: query
        type: List[str]
        description: "A list of search queries for web research."
      - name: rationale
        type: str
        description: "A brief explanation of why these queries are relevant."
  - name: Reflection
    description: "Analysis of research sufficiency and identified knowledge gaps."
    fields:
      - name: is_sufficient
        type: bool
        description: "True if the summaries are sufficient to answer the question."
      - name: knowledge_gap
        type: str
        description: "Description of missing information or areas for clarification."
      - name: follow_up_queries
        type: List[str]
        description: "A list of follow-up queries to address the knowledge gap."
```

#### Example Generated Code (`/backend_gen/agent/tools_and_schemas.py`):
```python
# This file is auto-generated from /tasks/artifacts/tools_spec.yaml.
# Do not edit this file directly.

from typing import List
from pydantic import BaseModel, Field

class SearchQueryList(BaseModel):
    """A structured list of web search queries and the rationale."""
    query: List[str] = Field(description="A list of search queries for web research.")
    rationale: str = Field(description="A brief explanation of why these queries are relevant.")

class Reflection(BaseModel):
    """Analysis of research sufficiency and identified knowledge gaps."""
    is_sufficient: bool = Field(description="True if the summaries are sufficient to answer the question.")
    knowledge_gap: str = Field(description="Description of missing information or areas for clarification.")
    follow_up_queries: List[str] = Field(description="A list of follow-up queries to address the knowledge gap.")
```
#### Example Output Documentation (`/docs/instructions/03_validate_tools_spec.md`):
```md
# How to Validate and Generate Tools & Schemas

The `tools_spec.yaml` defines the Pydantic models for structured LLM outputs. This process validates the spec and generates the corresponding Python code.

### Steps:
1.  Run the generation command:
    ```bash
    python backend_gen/validation/run_validations.py --generate-tools
    ```
2.  **Expected Outcome:** Success. A file will be created/updated at `/backend_gen/agent/tools_and_schemas.py`.
3.  **Manual Review:** Does the generated code contain the correct Pydantic models?
```
#### ✅ Expanded Validation Checklist:
1.  **File Existence:** `assert os.path.isfile("./tasks/artifacts/tools_spec.yaml")` -> PASS/FAIL
2.  **YAML Parsing & Schema Check:** Load YAML and validate schema for each item in `schemas`.
3.  **Code Generation Check:** `assert os.path.isfile("./backend_gen/agent/tools_and_schemas.py")` -> PASS/FAIL
4.  **Linting:** Run `black` and `ruff` on the generated file. -> FAIL on errors.
5.  **Type Checking:** Run `mypy` on the generated file. -> FAIL on errors.

---

*Phases for **Prompts**, **Node Implementation**, and **Graph Assembly** follow the same explicit, detailed pattern.*

### Phase 5: Node Implementation

**Objective:** To write the Python business logic for each node and create unit tests.

#### Task Summary:
1.  **Input:** Read all specification files in `/tasks/artifacts/`.
2.  **Output:** Create Python files for each node in `/backend_gen/agent/nodes/`.
3.  **Output:** Create corresponding unit test files in `/backend_gen/tests/nodes/`.
4.  **Output:** Create `/docs/instructions/05_unit_test_nodes.md`.

#### Example Generated Code (`/backend_gen/agent/nodes/reflection_node.py`):
```python
# Implementation for the 'reflection' node.
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import OverallState
from agent.tools_and_schemas import Reflection
from agent.prompts import reflection_instructions # Assuming prompts are loaded into a module

def reflection(state: OverallState) -> dict:
    """Analyzes search results and decides if more research is needed."""
    
    print("---REFLECTING ON RESEARCH---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(Reflection)
    
    research_summary = "\n\n---\n\n".join(state["web_research_result"])
    
    prompt = reflection_instructions.format(
        research_topic=state["messages"][-1].content,
        summaries=research_summary
    )
    
    reflection_result = structured_llm.invoke(prompt)
    
    loop_count = state.get("research_loop_count", 0) + 1
    
    return {
        "is_sufficient": reflection_result.is_sufficient,
        "knowledge_gap": reflection_result.knowledge_gap,
        "follow_up_queries": reflection_result.follow_up_queries,
        "research_loop_count": loop_count,
    }
```
#### Example Output Documentation (`/docs/instructions/05_unit_test_nodes.md`):
```md
# How to Unit Test Agent Nodes

Each node is a self-contained function and MUST be tested in isolation using `pytest`.

### Steps:

1.  For each node file in `/backend_gen/agent/nodes/`, ensure a corresponding test file exists in `/backend_gen/tests/nodes/`.
2.  The test should mock any external calls (like LLMs) and assert the node's output dictionary.
3.  Run all unit tests from the project root:
    ```bash
    pytest backend_gen/tests/
    ```
4.  **Expected Outcome:** All tests should pass. The output should show `X passed in Ys`.
```
#### ✅ Expanded Validation Checklist:
1.  **File Existence:** For each `node` in `agent_spec.yaml`, `assert os.path.isfile(f"./backend_gen/agent/nodes/{node['name']}_node.py")` -> PASS/FAIL
2.  **Test Existence:** For each node file, `assert os.path.isfile(f"./backend_gen/tests/nodes/test_{node['name']}_node.py")` -> PASS/FAIL
3.  **Test Execution:** Run `pytest backend_gen/tests/`. The process must exit with code 0. -> FAIL otherwise.
4.  **Test Coverage (Advanced):** Run `pytest --cov=backend_gen/agent/nodes`. Assert that the coverage percentage is above a defined threshold (e.g., 85%). -> FAIL if below threshold.

---

### Phase 6: Graph Assembly & Final Testing

**Objective:** To dynamically build the full graph from specs and provide end-to-end testing instructions.

#### Example Generated Code (`/backend_gen/agent/graph_builder.py`):
```python
# This file reads the agent_spec.yaml to dynamically build the graph.
import yaml
from langgraph.graph import StateGraph, START, END
from agent.state import OverallState
# Dynamically import node functions
from agent.nodes import generate_query, web_research, reflection, finalize_answer
# Dynamically import conditional edge functions
from agent.conditional_edges import continue_to_web_research, evaluate_research

def build_graph():
    """Builds the agent graph from the specification file."""
    
    with open("./tasks/artifacts/agent_spec.yaml", 'r') as f:
        spec = yaml.safe_load(f)

    # Map string names to actual Python functions
    node_map = {
        "generate_query": generate_query,
        "web_research": web_research,
        "reflection": reflection,
        "finalize_answer": finalize_answer,
    }
    conditional_edge_map = {
        "continue_to_web_research": continue_to_web_research,
        "evaluate_research": evaluate_research,
    }

    builder = StateGraph(OverallState)

    # Add all nodes from the spec
    for node_spec in spec['nodes']:
        builder.add_node(node_spec['name'], node_map[node_spec['name']])

    # Set the entry point
    builder.add_edge(START, spec['entry_point'])

    # Add all standard edges
    for edge_spec in spec.get('edges', []):
        builder.add_edge(edge_spec['from'], edge_spec['to'])

    # Add all conditional edges
    for cond_edge_spec in spec.get('conditional_edges', []):
        condition_func = conditional_edge_map[cond_edge_spec['condition_func']]
        path_map = {path: path for path in cond_edge_spec['paths']}
        builder.add_conditional_edges(cond_edge_spec['from'], condition_func, path_map)

    return builder.compile()
```
#### Example Output Documentation (`/docs/instructions/99_end_to_end_agent_testing.md`):
```md
# End-to-End Agent Testing Guide

This guide explains how to run and test the fully assembled agent.

### 1. Visualize the Graph

Verify the compiled graph matches the specification.
```bash
python backend_gen/validation/run_validations.py --visualize-graph
# An image file named graph.png will be saved in the root directory.
```

### 2. Run the Agent via API

1.  Start the FastAPI server: `uvicorn backend_gen.app:app --reload`
2.  In a new terminal, send a request using `curl`:

    ```bash
    curl -N -X POST "http://127.0.0.1:8000/agent/stream" \
         -H "Content-Type: application/json" \
         -d '{
               "input": {
                 "messages": [{"type": "human", "content": "What are the latest advancements in LLM agent memory?"}]
               }
             }'
    ```
3. **Expected Outcome:** A stream of JSON chunks representing the agent's execution flow.
```
#### ✅ Expanded Validation Checklist:
1.  **Graph Compilation:** The `build_graph()` function must execute without errors.
2.  **Visualization Check:** The `--visualize-graph` command must successfully create a non-empty `graph.png` file.
3.  **API Health Check:**
    *   Start the `uvicorn` server as a background process.
    *   Make a GET request to a `/health` endpoint on the app. `assert response.status_code == 200`. -> FAIL if not.
    *   Shut down the server process.
4.  **API Smoke Test:**
    *   Start the server.
    *   Run the `curl` command from the documentation. `assert response.status_code == 200`.
    *   Check that the streamed response is valid, non-empty JSON. -> FAIL otherwise.
    *   Shut down the server.