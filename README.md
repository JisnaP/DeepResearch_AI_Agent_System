# Deep Research AI Agentic System

An implementation of a multi-agent research system using LangChain and LangGraph. This system automates the research and drafting process by searching the web, analyzing search results, drafting a comprehensive answer, and refining it to a final answer.

## Features
- **Web Search**: Searches the web using the Tavily search tool.
- **Research Analysis**: Evaluates whether the gathered research answers the query and generates follow-up questions if needed.
- **Follow-up Research**: Conducts additional searches based on follow-up questions to gather more relevant information.
- **Drafting**: Synthesizes the research into a comprehensive draft that answers the original query.
- **Evaluation**: Assesses the quality of the drafted answer and identifies if further research is needed.
- **Finalization**: Refines the draft into a polished final answer with proper citations.

## Prerequisites



You will need the following API keys:
- OpenAI API Key
- Tavily API Key

These keys should be stored in a `.env` file in the root directory.


## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/JisnaP/deep-research-ai-agent.git
cd deep-research-ai-agent
```
### 2. Create and activate a virtual environment:

```bash
conda create -p ./venv python=3.10 -y
conda activate ./venv
```
### 3. Install dependencies:
```bash
pip install -r requirements.txt
```
### 4. Set up your environment variables:
```bash
OPENAI_API=<your_openai_api_key>
TAVILY_API=<your_tavily_api_key>

```
### Usage

```bash
python app.py

```
### To run in asynchronous mode

```bash
python asynchronous_deepsearch_agent.py

```
This will start the research agent, which will:

1. Search the web based on the query.

2. Analyze the research results and check if further research is needed.

3. Generate a draft based on the research.

4. Evaluate the quality of the draft and refine it.

5. Return the final answer.

### Example Usage
```bash
user_query = "What are the latest advancements in quantum computing and their potential impact on cryptography?"
answer = asyncio.run(research_agent_system(user_query))
print(answer)

```
### Example answer 

### License

This project is licensed under the MIT License.