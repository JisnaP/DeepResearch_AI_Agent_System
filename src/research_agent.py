import os
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()
OPENAI_API=os.getenv("OPENAI_API")
TAVILY_API=os.getenv("TAVILY_API")
# Configure API keys
os.environ["OPENAI_API_KEY"] = OPENAI_API
os.environ["TAVILY_API_KEY"] = TAVILY_API

current_date = datetime.now().strftime("%Y-%m-%d")
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

# Initialize LLM models for the agents
researcher_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# Initialize the Tavily search tool
search_tool = TavilySearch(k=8)

# Research Agent Implementation
def search_web(state: AgentState) -> AgentState:
    """Search the web for information related to the query."""
    print("🔍 Research Agent: Searching the web...")
    
    # Use Tavily to search for information
    search_results = search_tool.invoke(state.query)
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

def analyze_research_needs(state: AgentState) -> AgentState:
    """Analyze the research results and determine if more research is needed."""
    print("🔍 Research Agent: Analyzing research needs...")
    

    research_analyzer_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Today's date is {current_date}.You are a research analyst who evaluates search results.
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
    
    result = chain.invoke({
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

def conduct_follow_up_research(state: AgentState) -> AgentState:
    """Conduct additional research based on follow-up questions."""
    print("🔍 Research Agent: Conducting follow-up research...")
    
    if not state.follow_up_questions:
        state.research_complete = True
        return state
    
    # Take the next follow-up question
    follow_up_query = state.follow_up_questions.pop(0)
    
    # Use Tavily to search for additional information
    additional_results = search_tool.invoke(follow_up_query)
    
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