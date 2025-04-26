# Deep Research AI Agentic System
# An implementation of a multi-agent research system using LangChain and LangGraph
import asyncio
import os
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END

if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
# Load environment variables
load_dotenv()
OPENAI_API=os.getenv("OPENAI_API")
TAVILY_API=os.getenv("TAVILY_API")
# Configure API keys
os.environ["OPENAI_API_KEY"] = OPENAI_API
os.environ["TAVILY_API_KEY"] = TAVILY_API

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

# Initialize LLM models for our agents
researcher_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
drafter_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# Initialize the Tavily search tool
search_tool = TavilySearch(k=8)

# Research Agent Implementation
async def search_web(state: AgentState) -> AgentState:
    """Search the web for information related to the query."""
    print("🔍 Research Agent: Searching the web...")
    
    # Use Tavily to search for information
    search_results = await search_tool.ainvoke(state.query)
    formatted_results = []
    
    # Clean up and filter results
    for r in search_results:
        if isinstance(r, dict):
            # Clean up content
            if "content" in r:
                # Remove obvious repeats and truncate very long content
                content = r["content"]
                if len(content) > 2000:
                    # Take only first 2000 chars to avoid repetition
                    content = content[:2000]
                r["content"] = content
            formatted_results.append(r)
    # Update the state with search results
    state.research_results.extend(formatted_results)
    return state

async def analyze_research_needs(state: AgentState) -> AgentState:
    """Analyze the research results and determine if more research is needed."""
    print("🔍 Research Agent: Analyzing research needs...")
    

    research_analyzer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research analyst who evaluates search results.
        Analyze the search results and determine if they adequately address the query.
        If not, generate follow-up questions that would help gather more relevant information."""),
        ("user", "Original Query: {query}"),
        ("user", "Research Results:{research_results}"),
        MessagesPlaceholder(variable_name="history"),
    ])
    
    chain = (
        research_analyzer_prompt 
        | researcher_llm 
        | StrOutputParser()
    )
    
    result = await chain.ainvoke({
        "query": state.query,
        "research_results":state.research_results ,
        "history": []
    })
    
    # Parse the result to determine if more research is needed
    if "FOLLOW-UP QUESTIONS:" in result:
        state.needs_more_research = True
        # Extract follow-up questions
        questions_part = result.split("FOLLOW-UP QUESTIONS:")[1]
        # Parse numbered or bullet-point questions
        questions = [q.strip() for q in questions_part.split("\n") if q.strip() and any(c.isdigit() for c in q[:2])]
        if not questions:  # Try another parsing approach if the above didn't work
            questions = [q.strip().strip('•-') for q in questions_part.split("\n") if q.strip() and (q.strip().startswith('•') or q.strip().startswith('-'))]
        if not questions:  # If still no structured questions, take whole section
            questions = [questions_part.strip()]
        
        state.follow_up_questions.extend(questions)
    else:
        state.research_complete = True
    
    return state

async def conduct_follow_up_research(state: AgentState) -> AgentState:
    """Conduct additional research based on follow-up questions."""
    print("🔍 Research Agent: Conducting follow-up research...")
    
    if not state.follow_up_questions:
        state.research_complete = True
        return state
    
    # Take the next follow-up question
    follow_up_query = state.follow_up_questions.pop(0)
    
    # Use Tavily to search for additional information
    additional_results = await search_tool.ainvoke(follow_up_query)
    
    # Clean up and filter results
    formatted_results = []
    for r in additional_results:
        if isinstance(r, dict):
            # Add context about which question these results address
            r["follow_up_query"] = follow_up_query
            
            # Clean up content
            if "content" in r:
                # Remove obvious repeats and truncate very long content
                content = r["content"]
                if len(content) > 2000:
                    # Take only first 2000 chars to avoid repetition
                    content = content[:2000]
                r["content"] = content
            formatted_results.append(r)

    # Update the state with new search results
    state.research_results.extend(formatted_results)
    
    # If there are no more follow-up questions, mark research as complete
    if not state.follow_up_questions:
        state.research_complete = True
        state.needs_more_research = False
    
    return state

# Drafting Agent Implementation
async def draft_answer(state: AgentState) -> AgentState:
    """Draft an answer based on the research results."""
    print("✍️ Drafting Agent: Creating initial draft...")
     # Deduplicate research results based on content
    seen_content = set()
    filtered_results = []
    for result in state.research_results:
        content = result.get("content", "")
        content_hash = hash(content[:100])  # Use first 100 chars as a signature
        if content_hash not in seen_content and content.strip():
            seen_content.add(content_hash)
            filtered_results.append(result)
  
    # Format research results with citations (including actual URLs)
    formatted_research_results = []
    for i, result in enumerate(filtered_results):
        content = result.get("content", "")
        url = result.get("url", "")
        citation = f"[{i+1}]({url})" if url else f"[{i+1}]"
        formatted_research_results.append(f"{content} {citation}")
    
    formatted_research_results_text = "\n".join(formatted_research_results)
    drafter_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at synthesizing research into clear, comprehensive answers.
        Based on the provided research results, create a well-structured and informative response that directly addresses the original query.
        At the end of each paragraph or key point, include a citation in this format: [source](URL).
        If multiple results support a point, include up to 2 citations.
        Do not invent citations not present in the list below.
        If the research results don't contain enough information to fully answer the query, note this in your response."""),
        ("user", "Original Query: {query}"),
        ("user", "Research Results:{formatted_research_results_text}"),
    ])
    
    chain = (
        drafter_prompt 
        | drafter_llm 
        | StrOutputParser()
    )
    
    drafted_answer = await chain.ainvoke({
        "query": state.query,
        "formatted_research_results_text": formatted_research_results_text
    })
    
    state.drafted_answer = drafted_answer
    return state

async def evaluate_draft(state: AgentState) -> AgentState:
    """Evaluate the draft to determine if more research is needed."""
    print("✍️ Drafting Agent: Evaluating draft quality...")
    

    evaluator_prompt = ChatPromptTemplate.from_messages([
        ("system", """You evaluate the quality and completeness of an answer draft.
        Determine if the draft adequately addresses the original query or if more research is needed.
        If certain aspects of the query remain unaddressed or if the information seems insufficient,
        indicate what additional information would be helpful."""),
        ("user", "Original Query: {query}"),
        ("user", "Drafted Answer: {drafted_answer}"),
        ("user", "Research Results:{research_results}"),
    ])
    
    chain = (
        evaluator_prompt 
        | researcher_llm 
        | StrOutputParser()
    )
    
    evaluation = await chain.ainvoke({
        "query": state.query,
        "drafted_answer": state.drafted_answer,
        "research_results": state.research_results
    })
    
    # Determine if more research is needed based on the evaluation
    if "ADDITIONAL RESEARCH NEEDED:" in evaluation:
        state.needs_more_research = True
        
        # Extract follow-up questions
        questions_part = evaluation.split("ADDITIONAL RESEARCH NEEDED:")[1]
        questions = [q.strip() for q in questions_part.split("\n") if q.strip() and not q.strip().startswith("•")]
        
        state.follow_up_questions.extend(questions)
    else:
        state.needs_more_research = False
    
    return state

async def finalize_answer(state: AgentState) -> AgentState:
    """Finalize the answer by refining the draft."""
    print("✍️ Drafting Agent: Finalizing answer...")
    unique_results = {}
    for result in state.research_results:
        url = result.get("url", "")
        if url and url not in unique_results:
            unique_results[url] = result
    
    # If no URLs were found, use the whole list
    if not unique_results:
        unique_results = {i: result for i, result in enumerate(state.research_results)}
    # Format the unique results
    formatted_research_results = []
    for i, (_, result) in enumerate(unique_results.items()):
        content = result.get("content", "")
        url = result.get("url", "")
        citation = f"[{i+1}]({url})" if url else f"[{i+1}]"
        formatted_research_results.append(f"{content} {citation}")
    
    formatted_research_results_text = "\n".join(formatted_research_results)

    finalizer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a skilled editor who refines draft content into polished, final answers.
        Review the drafted answer and make improvements to:
        1. Ensure all parts of the original query are addressed
        2. Improve clarity, structure, and flow
        3. Eliminate redundancy
        4. At the end of each key point, add a citation in format like [1], [2], etc., referring to the corresponding research source.
        5. Format the answer appropriately with headers, bullet points, etc. as needed
        6. Include a complete and properly formatted References section at the end with all cited sources
        
        IMPORTANT: Make sure all citations in the text have corresponding entries in the References section.
        Format the References section like this:
        
        References:
        1. [Title 1](URL1)
        2. [Title 2](URL2)
        Ensure every citation in the text has a corresponding entry in the References list.
        Do not remove any citations from the body.
        """),
        ("user", "Original Query: {query}"),
        ("user", "Draft Answer: {drafted_answer}"),
        ("user", "Research Results:{formatted_research_results_text}"),
    ])
    
    chain = (
        finalizer_prompt 
        | drafter_llm 
        | StrOutputParser()
    )
    
    final_answer = await chain.ainvoke({
        "query": state.query,
        "drafted_answer": state.drafted_answer,
        "formatted_research_results_text": formatted_research_results_text
    })
    
    state.final_answer = final_answer
    return state

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
async def research_agent_system(query: str) -> str:
    """Main function to execute the research agent system."""
    print(f"🚀 Starting research on: {query}")
    
    # Create the workflow graph
    graph = create_research_graph()
    
    # Compile the graph into a runnable
    app = graph.compile()
    
    # Run the workflow
    initial_state = AgentState(query=query)
    result = await app.ainvoke(initial_state)
    
    print("\n✅ Research complete!")
    return result['final_answer']

# usage
if __name__ == "__main__":
    user_query = "What are the latest advancements in quantum computing and their potential impact on cryptography?"
    answer = asyncio.run(research_agent_system(user_query))
    print("\n📝 FINAL ANSWER:")
    print(answer)
