import os
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from src.research_agent import search_web,analyze_research_needs,conduct_follow_up_research
from src.draft_agent import draft_answer,evaluate_draft,finalize_answer



# Define the state schema for our agent system
class AgentState(BaseModel):
    """State for the research agent system."""
    query: str
    research_results: List[Dict[str, Any]] = []
    follow_up_questions: List[str] = []
    drafted_answer: str = ""
    final_answer: str = ""
    needs_more_research: bool = False
    research_complete: bool = False

# Orchestration with LangGraph
def create_research_graph() -> StateGraph:
    """Create the LangGraph for orchestrating the research workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("search_web", search_web)
    workflow.add_node("analyze_research_needs", analyze_research_needs)
    workflow.add_node("conduct_follow_up_research", conduct_follow_up_research)
    workflow.add_node("draft_answer", draft_answer)
    workflow.add_node("evaluate_draft", evaluate_draft)
    workflow.add_node("finalize_answer", finalize_answer)
    
    # Define the flow
    workflow.add_edge("search_web", "analyze_research_needs")
    workflow.add_conditional_edges(
        "analyze_research_needs",
        lambda state: "conduct_follow_up_research" if state.needs_more_research else "draft_answer"
    )
    workflow.add_conditional_edges(
        "conduct_follow_up_research",
        lambda state: "analyze_research_needs" if not state.research_complete else "draft_answer"
    )
    workflow.add_edge("draft_answer", "evaluate_draft")
    workflow.add_conditional_edges(
        "evaluate_draft",
        lambda state: "conduct_follow_up_research" if state.needs_more_research else "finalize_answer"
    )
    workflow.add_edge("finalize_answer", END)
    
    # Set entry point
    workflow.set_entry_point("search_web")
    
    return workflow

# 4. Main Application
def research_agent_system(query: str) -> str:
    """Main function to execute the research agent system."""
    print(f"🚀 Starting research on: {query}")
    
    # Create the workflow graph
    graph = create_research_graph()
    
    # Compile the graph into a runnable
    app = graph.compile()
    
    # Run the workflow
    initial_state = AgentState(query=query)
    result = app.invoke(initial_state)
    
    print("\n✅ Research complete!")
    return result['final_answer']

# usage
if __name__ == "__main__":
    user_query = "What are the latest advancements in quantum computing and their potential impact on cryptography?"
    answer = research_agent_system(user_query)
    print("\n📝 FINAL ANSWER:")
    print(answer)
