# Backend Code Blueprint

## Core Components Implementation

### 1. Configuration

The agent's configuration is handled by a Pydantic model in `configuration.py`:

```python
class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="gemini-2.0-flash",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="gemini-2.5-flash-preview-04-17",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="gemini-2.5-pro-preview-05-06",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )
```

### 2. State Management

State types are defined in `state.py` using TypedDict:

```python
class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str

class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int
```

### 3. Tools and Schemas

Tool schemas are defined using Pydantic models in `tools_and_schemas.py`:

```python
class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )

class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )
```

### 4. Graph Workflow Implementation

The agent's workflow is implemented in `graph.py` using LangGraph nodes:

```python
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question."""
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    formatted_prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}

def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the Google Search API."""
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, 
        state["id"]
    )
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }
```

### 5. Utility Functions

Helper functions in `utils.py` for handling citations and URLs:

```python
def resolve_urls(urls_to_resolve: List[Any], id: int) -> Dict[str, str]:
    """Create a map of vertex ai search urls to short urls with unique ids."""
    prefix = f"https://vertexaisearch.cloud.google.com/id/"
    urls = [site.web.uri for site in urls_to_resolve]

    resolved_map = {}
    for idx, url in enumerate(urls):
        if url not in resolved_map:
            resolved_map[url] = f"{prefix}{id}-{idx}"

    return resolved_map

def insert_citation_markers(text, citations_list):
    """Inserts citation markers into text based on start and end indices."""
    sorted_citations = sorted(
        citations_list, 
        key=lambda c: (c["end_index"], c["start_index"]), 
        reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text
```

### 6. FastAPI Application

The web API is implemented in `app.py`:

```python
app = FastAPI()

def create_frontend_router(build_dir="../frontend/dist"):
    """Creates a router to serve the React frontend."""
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}."
        )
        async def dummy_frontend(request):
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )
        return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)

# Mount the frontend under /app
app.mount("/app", create_frontend_router(), name="frontend")
```

## Usage Examples

### 1. Basic Agent Usage

```python
from agent import graph

# Initialize conversation
state = graph.invoke({
    "messages": [
        {"role": "user", "content": "Who won the euro 2024"}
    ],
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

### 2. Adding New Tools

To add a new tool:

1. Define the tool schema:
```python
class NewTool(BaseModel):
    """Schema for the new tool."""
    input_field: str = Field(description="Description of the input")
    output_field: str = Field(description="Description of the output")
```

2. Add tool implementation in graph.py:
```python
def new_tool_node(state: OverallState, config: RunnableConfig) -> OverallState:
    """New LangGraph node implementation."""
    configurable = Configuration.from_runnable_config(config)
    # Tool implementation
    return {
        "result": "tool_result",
        "state_update": "update_value"
    }
```

3. Register in workflow:
```python
workflow.add_node("new_tool", new_tool_node)
workflow.add_edge("previous_node", "new_tool")
workflow.add_edge("new_tool", "next_node")
```

## Error Handling

Example error handling implementation:

```python
def safe_web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """Web research with error handling."""
    try:
        return web_research(state, config)
    except Exception as e:
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [f"Error during research: {str(e)}"],
        }
```

## Testing

Example test structure:

```python
def test_generate_query():
    """Test query generation."""
    state = {
        "messages": [{"role": "user", "content": "test query"}],
        "initial_search_query_count": 3
    }
    config = {"configurable": {"query_generator_model": "test-model"}}
    
    result = generate_query(state, config)
    assert "query_list" in result
    assert len(result["query_list"]) == state["initial_search_query_count"]

def test_web_research():
    """Test web research functionality."""
    state = {
        "search_query": "test search",
        "id": "1"
    }
    config = {"configurable": {"query_generator_model": "test-model"}}
    
    result = web_research(state, config)
    assert "sources_gathered" in result
    assert "web_research_result" in result
