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
git clone https://github.com/JisnaP/DeepResearch_AI_Agent_System.git
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
### Example Answer
```bash
🚀 Starting research on: What are the latest advancements in quantum computing and their potential impact on cryptography?
🔍 Research Agent: Searching the web...
🔍 Research Agent: Analyzing research needs...
✍️ Drafting Agent: Creating initial draft...
✍️ Drafting Agent: Evaluating draft quality...
✍️ Drafting Agent: Finalizing answer...

✅ Research complete!

📝 FINAL ANSWER:
### Latest Advancements in Quantum Computing

Quantum computing has seen several significant advancements recently, which are crucial for the development of more robust quantum systems:

- **Stability of Qubits**: Researchers have made strides in enhancing the stability of qubits, the core elements of quantum computers. Improvements in qubit coherence times mean that these systems can perform complex computations for longer periods without errors [1].

- **Quantum Error Correction**: Progress in quantum error correction techniques is vital for creating reliable quantum computers. These advancements help in mitigating errors that naturally occur in quantum computations, thus paving the way for more accurate and scalable quantum computing systems [2].

### Impact on Cryptography

The rise of quantum computing presents profound implications for cryptography, particularly affecting systems based on public-key cryptography:

- **Vulnerability of Current Systems**: Quantum computers could potentially break cryptographic algorithms such as RSA and ECC (Elliptic Curve Cryptography), which are foundational to securing digital communications and data. This vulnerability stems from quantum algorithms like Shor's algorithm, which can efficiently factor large integers and compute discrete logarithms, tasks that are computationally intensive for classical computers [3].

### Developments in Post-Quantum Cryptography

In response to the threats posed by quantum computing, the field of post-quantum cryptography is developing systems that are secure against both quantum and classical computational attacks:

- **NIST's Standardization Initiative**: The National Institute of Standards and Technology (NIST) is leading efforts to standardize post-quantum cryptographic algorithms. This initiative is in its final stages, evaluating several candidate algorithms for their security efficacy and performance [4].

### Quantum Key Distribution (QKD)

Quantum Key Distribution (QKD) represents a significant advancement in the realm of quantum cryptography:

- **Secure Key Distribution**: QKD utilizes principles of quantum mechanics to distribute encryption keys securely. The security of QKD is based on the laws of physics, rather than on computational complexity, making it a robust method for secure communications. This technology has already been implemented in sectors such as secure banking and government communications [5].

### Conclusion

These advancements in quantum computing and cryptography are reshaping the landscape of digital security. W exploration and analysis.

### References:

1. [Improvements in Qubit Coherence](https://quantum-journal.org)
2. [Advances in Quantum Error Correction](https://quantum-journal.org)
3. [Quantum Algorithms and Cryptography](https://quantum-journal.org)
4. [NIST Post-Quantum Cryptography Initiative](https://nist.gov)
5. [Applications of Quantum Key Distribution](https://quantum-journal.org)
```
### License

This project is licensed under the MIT License.