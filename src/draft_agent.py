import os
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# Load environment variables
load_dotenv()
OPENAI_API=os.getenv("OPENAI_API")
TAVILY_API=os.getenv("TAVILY_API")
# Configure API keys
os.environ["OPENAI_API_KEY"] = OPENAI_API
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
drafter_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    # Drafting Agent Implementation
def draft_answer(state: AgentState) -> AgentState:
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
        ("system", f"""Today's date is {current_date}.You are an expert at synthesizing research into clear, comprehensive answers.
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
    
    drafted_answer = chain.invoke({
        "query": state.query,
        "formatted_research_results_text": formatted_research_results_text
    })
    
    state.drafted_answer = drafted_answer
    return state

def evaluate_draft(state: AgentState) -> AgentState:
    """Evaluate the draft to determine if more research is needed."""
    print("✍️ Drafting Agent: Evaluating draft quality...")
    

    evaluator_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Today's date is {current_date}.You evaluate the quality and completeness of an answer draft.
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
    
    evaluation = chain.invoke({
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

def finalize_answer(state: AgentState) -> AgentState:
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
    
    final_answer = chain.invoke({
        "query": state.query,
        "drafted_answer": state.drafted_answer,
        "formatted_research_results_text": formatted_research_results_text
    })
    
    state.final_answer = final_answer
    return state