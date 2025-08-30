# LangGraph Agents & Graphs

This repository contains a collection of AI agents and graph-based workflows built using [LangGraph](https://github.com/langchain-ai/langgraph). It demonstrates various agent architectures, conditional flows, looping graphs, and integration with LLMs (Mistral, OpenAI) for document processing and conversational AI.

## Directory Structure

```
.
├── agents/
│   ├── AIML Industry Analysis.pdf
│   ├── ChatBot.py
│   ├── ChatBot2.py
│   ├── Drafter.py
│   ├── Medical_Document_Input.pdf
│   ├── RagAgent.py
│   ├── ReActAgent.py
│   └── chroma.sqlite3
├── Graphs/
│   ├── Graph1.ipynb
│   ├── Graph2.ipynb
│   ├── Graph3.ipynb
│   ├── Graph4.ipynb
│   ├── Graph5.ipynb
│   └── ExerciseForGraph5.ipynb
├── requirements.txt
├── .env
├── .gitignore
└── conversation.txt
```

## Contents

### Agents

- **ChatBot.py / ChatBot2.py**: Conversational agents using Mistral/OpenAI models, with memory and conversation logging ([agents/ChatBot.py](agents/ChatBot.py), [agents/ChatBot2.py](agents/ChatBot2.py)).
- **Drafter.py**: Document agent for updating and saving documents via LLM and custom tools ([agents/Drafter.py](agents/Drafter.py)).
- **RagAgent.py**: Retrieval-Augmented Generation agent for PDF ingestion, chunking, and vector search using Chroma ([agents/RagAgent.py](agents/RagAgent.py)).
- **ReActAgent.py**: Implements the ReAct agent pattern (file present, see code for details).
- **PDFs**: Example documents for agent processing.

### Graphs

- **Graph1.ipynb**: Hello World agent with a simple greeting node ([Graphs/Graph1.ipynb](Graphs/Graph1.ipynb)).
- **Graph2.ipynb**: Multiple input graph for sum/product operations ([Graphs/Graph2.ipynb](Graphs/Graph2.ipynb)).
- **Graph3.ipynb**: Sequential graph for personalized messages ([Graphs/Graph3.ipynb](Graphs/Graph3.ipynb)).
- **Graph4.ipynb**: Conditional graph with branching based on operations ([Graphs/Graph4.ipynb](Graphs/Graph4.ipynb)).
- **Graph5.ipynb**: Looping graph for repeated random number generation ([Graphs/Graph5.ipynb](Graphs/Graph5.ipynb)).
- **ExerciseForGraph5.ipynb**: Higher/Lower game using graph logic ([Graphs/ExerciseForGraph5.ipynb](Graphs/ExerciseForGraph5.ipynb)).

## Setup

1. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
2. **Configure API keys**:
    - Add your Mistral/OpenAI API keys to `.env` as required by agents.

3. **Run Agents**:
    - Example:  
      ```sh
      python agents/ChatBot.py
      ```

4. **Explore Graphs**:
    - Open any notebook in `Graphs/` with Jupyter or VS Code and run cells to visualize and interact with the graphs.

## Features

- Conversational AI with memory and logging
- Document editing and saving via LLM tools
- PDF ingestion and vector search (RAG)
- Graph-based workflows: sequential, conditional, looping
- Interactive games and data flows

## Visualization

Most notebooks include graph visualizations using Mermaid and display outputs for each workflow.

## License

See repository for license details.

---

**Note:** For more details, see individual files and notebooks linked above.
