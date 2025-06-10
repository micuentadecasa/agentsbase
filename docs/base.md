Backend Architecture Guide
Core Components
1. Graph Structure
The agent graph is defined in graph.py and uses LangGraph's StateGraph for orchestration. The graph consists of several nodes that work together in a defined flow:

2. State Management
The state is managed through TypedDict classes defined in state.py:

Node Details
1. Query Generation Node
2. Web Research Node
3. Reflection Node
4. Answer Finalization Node
Tools and Utilities
URL Resolution: resolve_urls function handles URL shortening and citation management

Citation Management: get_citations and insert_citation_markers handle citation processing

Prompt Templates: Defined in prompts.py for each node's interactions

Configuration: Configuration class manages model settings and API configurations

Creating New Graphs
To create new graphs following this pattern:

Define your state classes using TypedDict
Create your node functions that process the state
Set up your graph using StateGraph
Define edges and conditional transitions
Compile the graph with builder.compile()
The key is ensuring your state management and transitions are well-defined, and that each node properly processes and passes state to the next node in the chain.