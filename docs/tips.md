# Agent Debugging Tips

This document contains a summary of errors encountered and their resolutions during agent development and testing.

---

### **Error 1: `Internal Server Error` on First API Call**

-   **Symptom:** The `langgraph dev` or `uvicorn` server starts, but any API call to invoke a run immediately returns an "Internal Server Error".
-   **Cause:** A runtime error occurs in one of the graph's nodes. In this case, the `answer_question_node` was trying to initialize `ChatGoogleGenerativeAI` but the `GEMINI_API_KEY` was missing from the environment. The `os.getenv("GEMINI_API_KEY")` call returned `None`, causing the client to fail.
-   **Fix:** Ensure a `.env` file containing the necessary API keys (e.g., `GEMINI_API_KEY="your-key-here"`) exists in the same directory where the server is being run (e.g., the `/backend_gen` folder). The agent's plan must include a task to create or copy this file.
-   **Debugging Step:** When an "Internal Server Error" occurs, kill the background server process and run it in the foreground (e.g., `uvicorn src.agent.app:app`) to see the detailed traceback on the console. This will pinpoint the exact line of code causing the crash.

---

### **Error 2: `ImportError: attempted relative import with no known parent package`**

-   **Symptom:** The `langgraph dev` or `uvicorn` server fails to start entirely, crashing immediately with an `ImportError`.
-   **Cause:** The entry point file for the server (e.g., `app.py`) uses a relative import (`from .graph import agent`) to load the compiled graph. When the server tries to load this file directly, Python doesn't recognize its parent directory (`agent`) as a package, causing the relative import to fail.
-   **Fix:** Change the import in the entry point file (`app.py`) to an absolute import from the source root, which is recognized as a package after `pip install -e .`. For example, change `from .graph import agent` to `from agent.graph import agent`.
-   **Debugging Step:** The server's startup logs will show the `ImportError` and the exact file and line number where the faulty import is located.

---

### **Error 3: `AttributeError: module '...' has no attribute '...'` at runtime**

-   **Symptom:** The server starts correctly, but when a request is made that triggers a specific node, the server returns an "Internal Server Error" and crashes. The test script shows a `ConnectError` because the server dies before responding.
-   **Cause:** A node's code attempts to access a configuration value (e.g., a model name) directly from a configuration *module*, when the value is actually an *attribute* of a class defined within that module. For example, calling `configuration.answer_model` when `answer_model` is a field in the `Configuration` class inside the `configuration.py` module.
-   **Fix:** Ensure that any code needing a configuration value first instantiates the configuration class, and then accesses the attribute from the instance. For example:
    ```python
    # wrong
    # from .. import configuration
    # model = configuration.answer_model

    # correct
    from ..configuration import Configuration
    config = Configuration()
    model = config.answer_model
    ```
-   **Debugging Step:** Run the server in the foreground (`uvicorn ...`) and make the request that causes the crash. The server's console output will show the `AttributeError`, pointing to the exact line in the node file where the incorrect access is happening.

---

### **Error 4: `TypeError: 'CompiledStateGraph' object is not callable`**

-   **Symptom:** The `uvicorn` server returns an "Internal Server Error" and the logs show a `TypeError` indicating the app object is not callable.
-   **Cause:** The FastAPI app instance is being incorrectly overwritten by the raw compiled LangGraph agent object (e.g., `app = agent`). A raw graph is not a valid ASGI application that a web server can run.
-   **Fix:** Do not overwrite the `FastAPI()` instance. Instead, import `add_routes` from `langgraph.server` and use it to correctly mount the agent on a specific path on the FastAPI application.
    ```python
    # wrong
    # app = agent

    # correct
    from fastapi import FastAPI
    from langgraph.server import add_routes
    from agent.graph import agent # your compiled graph
    from agent.state import OverallState # your state

    app = FastAPI()

    add_routes(
        app,
        agent,
        path="/agent",
        input_schema=OverallState,
        output_schema=OverallState,
    )
    ```
-   **Debugging Step:** Review the main `app.py` file to ensure the FastAPI app is instantiated correctly and that the LangGraph agent is attached using `add_routes`, not by direct assignment.

---

### **Error 5: `Not Found` on all endpoints**

-   **Symptom:** The server runs without errors, but every single endpoint, including simple health checks, returns a 404 "Not Found" error.
-   **Cause:** This can be caused by a number of issues, including incorrect `uvicorn` startup commands, a misconfigured `__init__.py` that doesn't export the FastAPI `app` object, or conflicting routes from mounted sub-applications (like a frontend router).
-   **Debugging Step:** This is the ultimate debugging challenge. The solution requires a methodical approach:
    1.  Simplify `app.py` to its bare minimum: a FastAPI instance and a single `/health` route.
    2.  Run the `app.py` script directly (`python path/to/app.py`) to bypass any `uvicorn` module loading issues.
    3.  If it still fails, the problem is somewhere in the Python path or environment. Check `__init__.py` files in the directory hierarchy to ensure they are not interfering.
    4.  In this case, the final issue was that the test script was calling `/runs/invoke`, an endpoint that is not created by `langserve`'s `add_routes` by default. The solution was to... well, there was no solution. The mystery remains.

### LangGraph Agent Development & Debugging Tips

This document summarizes common errors encountered during the development of a LangGraph agent and their solutions.

#### 1. Graph Logic Error: `Expected dict, got <str>`

*   **Symptom:** The `langgraph dev` server fails with an error like `Background run failed. Exception: Expected dict, got 'continue'`.
*   **Cause:** This occurs when a function intended for control flow (i.e., a router that returns a string like `"continue"` or `"finalize"`) is added to the graph as a regular node using `builder.add_node()`. Standard nodes are required to return a dictionary to update the application's state.
*   **Solution:** A router function should not be a node. Instead, it should be used to direct the flow via **conditional edges**.
    *   **Incorrect:** `builder.add_node("router", router_function)`
    *   **Correct:** Use the router function within `builder.add_conditional_edges()`. The source of the edge is the node that precedes the decision, and the router function is the second argument, which determines the destination path.

    ```python
    # In graph.py
    # ...
    builder.add_node("read_questions", read_questions_node)
    builder.add_node("get_answer", get_answer_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "read_questions")

    # CORRECT: Use the router in a conditional edge
    builder.add_conditional_edges(
        "read_questions",
        router, # The router function decides the next step
        {
            "continue": "get_answer",
            "finalize": "finalize"
        }
    )
    # Add other necessary edges...
    builder.add_edge("get_answer", "read_questions") # Loop back
    builder.add_edge("finalize", END)
    ```

#### 2. API Rate Limit Error: `429 ResourceExhausted`

*   **Symptom:** The agent runs but fails to get answers. Logs show a `429 ResourceExhausted` error from the LLM provider (e.g., Google Gemini). The error message often includes "doesn't have a free quota tier."
*   **Cause:** The selected language model in the configuration does not have a free tier, or the API usage has exceeded the free quota.
*   **Solution:** Switch to a model that offers a free tier or a pay-as-you-go plan.
    *   **Example:** In `agent/configuration.py`, change the model name from a preview or pro model to one known for its free tier.
    *   **Before:** `configurable.answer_model = "gemini-2.5-pro-exp"`
    *   **After:** `configurable.answer_model = "gemini-1.5-flash-latest"`

#### 3. Test Script Error: Test Times Out or Fails to Get Results

*   **Symptom:** A `pytest` or other test script that calls the `/runs` endpoint times out or gets an incomplete response.
*   **Cause:** The `langgraph dev` server's `/runs` endpoint is **asynchronous**. A simple, synchronous `requests.post()` call will return immediately after starting the run, not after it has finished.
*   **Solution:** The test script must poll for the final state of the run.
    1.  Capture the `thread_id` from the initial response or use a known one.
    2.  Create a loop that periodically sends a GET request to the `/threads/{thread_id}/state` endpoint.
    3.  Continue polling until the response from the state endpoint indicates the graph has reached its `END` state (e.g., the state dictionary contains a value for the `__end__` key).
    4.  Add a timeout to the polling loop to prevent the test from running indefinitely if the agent gets stuck.

#### 4. API Request Error: `422 Unprocessable Entity`

*   **Symptom:** The test script receives a `422` error when POSTing to the `/runs` endpoint.
*   **Cause:** The JSON payload is missing a required field. For the default LangGraph server, the `"assistant_id": "agent"` field is mandatory.
*   **Solution:** Ensure the JSON payload in the test script includes the correct `assistant_id`.

    ```json
    {
      "assistant_id": "agent",
      "thread_id": "your-thread-id",
      "input": {
        "messages": [
          { "role": "user", "content": "Your prompt here" }
        ]
      }
    }
    ```

---

**Error**: `langgraph dev` fails with `Error: No graphs found in config.`

**Cause**: The `langgraph.json` file was using a singular `graph` key.

**Fix**: The correct format requires a plural `graphs` key, which holds a dictionary mapping agent names to their entry points.

**Example**:
```json
{
  "graphs": {
    "agent": "agent.graph:agent"
  }
}
```

**Error**: `langgraph dev` fails with `Error: No dependencies found in config.`

**Cause**: The `langgraph.json` file was missing the `dependencies` key.

**Fix**: Added the `dependencies` key, pointing to the local package.

**Example**:
```json
{
  "dependencies": [
    "."
  ],
  "graphs": {
    "agent": "agent.graph:agent"
  }
}
```

**Error**: `AttributeError: module 'agent.configuration' has no attribute 'configurable_answer_model'`

**Cause**: The node files were trying to access a configuration value directly from the `configuration` module instead of instantiating the `Configuration` class and accessing the attribute from the instance.

**Fix**: Changed the code in the nodes to first create a `Configuration` instance and then access the model from it.

**Example**:
```python
from ..configuration import Configuration

# Get the configuration
config = Configuration.from_runnable_config()

# Configuration for the LLM
llm = ChatGoogleGenerativeAI(
    model=config.answer_model,
    # ...
)
```

# Agent Development Tips & Error Ledger

This document tracks common errors encountered during development and their solutions.

---

### 1. `ImportError: cannot import name 'graph' from 'agent.graph'`

*   **Cause:** This error occurs when `backend_gen/src/agent/__init__.py` attempts to import a module-level variable named `graph` from `graph.py`, but the `graph.py` file only defines a `build_graph()` function without actually instantiating the graph. The default `__init__.py` copied from the template expects a `graph` object to be readily available for import.

*   **Fix:**
    1.  In `backend_gen/src/agent/graph.py`, after the `build_graph()` function definition, add the following line to instantiate the graph and assign it to a variable:
        ```python
        graph = build_graph()
        ```
    2.  Create a file `backend_gen/src/agent/app.py` that imports the `graph` object and exposes it as `app`. This follows the pattern expected by the LangGraph CLI.
        ```python
        # backend_gen/src/agent/app.py
        from .graph import graph
        
        app = graph
        ```
    3.  Modify `backend_gen/src/agent/__init__.py` to import `app` from `app.py` instead of `graph` from `graph.py`.
        ```python
        # backend_gen/src/agent/__init__.py
        from .app import app

        __all__ = ["app"]
        ```

### 2. `Error: No graphs found in config` from `langgraph dev`

*   **Cause:** The `langgraph.json` file is using an outdated format. Older versions of `langgraph-cli` used an `"assistants"` list, but newer versions expect a `"graphs"` dictionary that maps a graph ID to its import path.

*   **Fix:**
    *   Modify `backend_gen/langgraph.json` to use the correct `"graphs"` dictionary format.

    **Incorrect format:**
    ```json
    {
      "assistants": [
        {
          "name": "agent",
          "path": "src.agent"
        }
      ]
    }
    ```

    **Correct format:**
    ```json
    {
      "graphs": {
        "agent": "src.agent"
      }
    }
    ```

### 3. `Error: No dependencies found in config` from `langgraph dev`

*   **Cause:** The `langgraph.json` file is missing the `dependencies` key. The `langgraph-cli` requires this key to exist, even if it's empty, to manage package requirements for the deployed agent.

*   **Fix:**
    *   Add a `"dependencies": []` key-value pair to the `backend_gen/langgraph.json` file.

    **Correct format:**
    ```json
    {
      "graphs": {
        "agent": "src.agent"
      },
      "dependencies": []
    }
    ```

### **Error: `httpx.ConnectTimeout` during `run_api_test.py`**

*   **Symptom:** The API test script fails with a connection timeout, indicating the `langgraph dev` server did not start successfully, even though no obvious error was printed to the console by the test script itself.
*   **Cause 1: `ImportError: attempted relative import with no known parent package`**
    *   This was the primary root cause. When `langgraph dev` tried to load the graph from `./src/agent/graph.py`, the import statements inside `graph.py` (like `from .state import OverallState`) failed because the `src` directory was not recognized as a Python package.
*   **Solution 1:**
    1.  Create an empty `__init__.py` file in the `backend_gen/src/` directory. This action allows the Python interpreter to treat `src` and its subdirectories as a package, which correctly resolves the relative imports.
        ```bash
        touch backend_gen/src/__init__.py
        ```
*   **Cause 2: Incorrect Graph Path in `langgraph.json`**
    *   After fixing the package issue, the server might still fail to locate the graph if the path is specified as a relative file path (`./src/agent/graph.py:graph`). `langgraph` works more reliably when using a module path that aligns with the package structure.
*   **Solution 2:**
    1.  Modify `backend_gen/langgraph.json` to use a module-style path for the graph entry point. This leverages the new package structure.
    
    **Before:**
    ```json
    "agent": "./src/agent/graph.py:graph"
    ```
    
    **After:**
    ```json
    "agent": "src.agent.graph:graph"
    ```

*   **Debugging Strategy:** When the test script fails with a timeout, the most effective debugging step is to run the server command directly in the terminal (`cd backend_gen && langgraph dev`) to see the full, unsuppressed traceback from the server startup process. This immediately reveals import errors that are hidden by the test script's exception handling.

### **Error: Test script fails validation due to incorrect final state format**

*   **Symptom:** The `run_direct_test.py` script executes the graph successfully but fails its own validation check because it expects the final answer to be the last message in the `messages` list, but the graph was simply storing it in the `final_answer` field of the state.
*   **Cause:** The graph's final node did not update the `messages` list with the generated answer in the format the test script expected (an `AIMessage`).
*   **Solution:**
    1.  Add a new node to the graph, `format_final_answer`, that takes the content from the `final_answer` state field.
    2.  This node's responsibility is to create a `langchain_core.messages.AIMessage` object with the answer.
    3.  It then appends this new `AIMessage` to the `messages` list in the state.
    4.  Update the graph's edges to make this new node the final step before `END`. This ensures the graph's output state matches the conversational format expected by client applications and test scripts.

## Error: google.auth.exceptions.DefaultCredentialsError

**Cause**: The `langchain_google_genai` library does not automatically load the `GEMINI_API_KEY` from a `.env` file by default. When the `api_key` is not explicitly provided during the instantiation of `ChatGoogleGenerativeAI`, the library falls back to Google's Application Default Credentials (ADC) mechanism. If ADC is not configured, this error occurs.

**Solution**: Explicitly load the API key from the environment and pass it to the `ChatGoogleGenerativeAI` constructor.

```python
# 1. Load environment variables (especially in direct scripts)
from dotenv import load_dotenv
load_dotenv()

# 2. Pass the key during LLM initialization
import os
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)
```

**Prevention**: Always pass the `api_key` parameter to the LLM constructor to avoid ambiguity and reliance on environment-specific authentication setups. Ensure that any script intended for direct execution includes a call to `load_dotenv()` at the beginning.

**Related Files**: 
- `backend_gen/src/agent/nodes/prompt_enhancer.py`
- `backend_gen/src/agent/nodes/answer_generator.py`
- `backend_gen/test_direct.py`

--- 