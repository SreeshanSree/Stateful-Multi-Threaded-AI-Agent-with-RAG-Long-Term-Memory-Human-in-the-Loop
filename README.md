# Autonomous Trading & RAG Assistant 🤖📈

An enterprise-grade, stateful ReAct (Reasoning and Acting) AI agent built with **LangGraph** and **Streamlit**. This application autonomously routes between live financial API lookups, web searches, and internal document retrieval (Agentic RAG), while enforcing strict **Human-in-the-Loop (HITL)** safety guardrails for transactional operations.

---

## 🚀 Key Features

* **Stateful ReAct Architecture:** Engineered a cyclic LangGraph agent (`chatbot` → `tools_condition` → `tools` → `chatbot`) that dynamically selects tools based on user intent.
* **Agentic RAG & Vector Database:** Integrated `FAISS` and Azure OpenAI Embeddings (`text-embedding-3-small`) to chunk and index uploaded PDFs on the fly using `PyMuPDF`. The agent autonomously calls `rag_tool` with thread-isolated document context when questions reference uploaded files.
* **Human-in-the-Loop (HITL) Safety Gate:** High-stakes transactional operations (such as `purchase_stock`) are paused mid-execution using LangGraph's native `interrupt()`. The graph freezes execution state and awaits explicit user confirmation (`"yes"` / `"no"`) before executing order placement.
* **Persistent Multi-Thread Memory:** Leveraged SQLite (`SqliteSaver`) as a checkpointer. The agent preserves full conversational history and graph checkpoints across multiple parallel chat sessions.
* **Decoupled UI & Dynamic Chat Titles:** Built an interactive Streamlit frontend with real-time response token streaming, multi-session management, and asynchronous title generation stored in a persistent SQLite table (`chat_titles`).
* **Live REST API Integration:** Fetches real-time market data directly via the AlphaVantage REST API, supplemented with DuckDuckGo live web search.

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Agentic Framework** | [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain Core](https://github.com/langchain-ai/langchain) |
| **Language Model & Embeddings** | Azure OpenAI (`gpt-4.1-mini`, `text-embedding-3-small`) |
| **Vector Store & RAG** | FAISS (`faiss-cpu`), `PyMuPDF` (`fitz`), `RecursiveCharacterTextSplitter` |
| **Memory & Persistence** | SQLite with LangGraph `SqliteSaver` checkpointer |
| **Frontend UI** | [Streamlit](https://streamlit.io/) |
| **External APIs & Tools** | AlphaVantage API (Market Data), DuckDuckGo Search |

---

## 🧠 System Architecture

The graph implements a stateful ReAct cycle with persistent checkpoints and interrupt gates:

```mermaid
flowchart TD
    Start([User Input in Streamlit]) --> LLM[Chatbot LLM Node<br>Azure OpenAI]
    LLM --> Condition{tools_condition}
    
    Condition -- "Informational Tool" --> Tools[ToolNode<br>RAG / Stock Price / Search]
    Tools --> LLM
    
    Condition -- "Transactional Tool<br>(purchase_stock)" --> Interrupt[HITL Gate: interrupt()<br>Freezes Execution State]
    Interrupt --> UserDecision{User Approval<br>yes / no}
    UserDecision -- "yes" --> OrderSuccess[Execute Trade & Return Confirmation]
    UserDecision -- "no" --> OrderCancelled[Cancel Trade & Return Decline Notice]
    OrderSuccess --> Tools
    OrderCancelled --> Tools
    
    Condition -- "No Tool Needed" --> StreamUI[Stream Response Chunks to Streamlit UI]
    StreamUI --> Checkpoint[(SQLite Checkpointer<br>SqliteSaver State)]
    Checkpoint --> End([Complete Turn])
```

### Flow Breakdown

1. **User Input:** Streamlit captures user prompts and passes thread identifiers (`thread_id`) to the compiled LangGraph runner.
2. **LLM Node:** Azure OpenAI evaluates conversational context against bound tools (`rag_tool`, `stock_price`, `purchase_stock`, `DuckDuckGoSearchRun`).
3. **Conditional Edge (`tools_condition`):**
   - **Informational tools (`rag_tool`, `stock_price`, search):** Execute automatically in `ToolNode` and route results back to the LLM node to synthesize an answer.
   - **Transactional tool (`purchase_stock`):** Triggers `interrupt()`, halting graph execution until the user submits a confirmation response.
4. **State Persistence:** Every checkpoint is saved to SQLite, enabling uninterrupted session resumption, rollbacks, and multi-threaded conversations.

---

## 📂 Project Structure

```
autonomous-trading-rag/
├── backend.py            # LangGraph state machine, tool definitions, RAG pipeline & checkpointer
├── frontend.py           # Streamlit application, thread manager, streaming handler & HITL approval
├── requirements.txt      # Project dependencies
├── .env.example          # Template for required environment variables
├── .gitignore            # Git exclusions (checkpoints, env, caches)
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SreeshanSree/autonomous-trading-rag.git
cd autonomous-trading-rag
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory (refer to `.env.example`):

```bash
cp .env.example .env
```

Populate the required credentials:

```ini
# Azure OpenAI Credentials
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini

# AlphaVantage API Key (Market Data)
INFOWAY_API_KEY=your_alphavantage_api_key

# (Optional) LangSmith Tracing & Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=Chatbot_project
```

### 5. Launch the Application

```bash
streamlit run frontend.py
```

Open your browser at `http://localhost:8501`.

---

## 💡 Usage Examples

### 1. Real-Time Stock Price Lookup
> **User:** "What is the current stock price of Apple (AAPL)?"  
> **Agent:** Queries AlphaVantage API via `stock_price` tool and returns latest daily metrics.

### 2. Human-in-the-Loop Stock Purchase
> **User:** "Buy 10 shares of TSLA."  
> **Agent:** Hits the HITL gate:  
> `Approve buying 10 shares of TSLA.`  
> **User:** "yes"  
> **Agent:** Resumes graph via `Command(resume="yes")` and confirms order placement.

### 3. Agentic Document RAG
1. Upload a PDF in the Streamlit sidebar (e.g., quarterly earnings report or research whitepaper).
2. The document is chunked with `RecursiveCharacterTextSplitter` and indexed into an in-memory `FAISS` vector store tied to the active `thread_id`.
3. Ask questions about the uploaded file:
   > **User:** "What was the company's net revenue according to the uploaded report?"  
   > **Agent:** Routes query to `rag_tool(query=..., thread_id=...)` and synthesizes an answer grounded in the document context.

---

## 🛡️ Security & Production Notes

- **Secrets Management:** Never commit `.env` or production credentials.
- **State Checkpointing:** SQLite checkpoint databases (`chatbot`, `*.db`) are excluded via `.gitignore` to prevent leaking conversational state into source control.
- **Transactional Safety:** Critical external mutations should always use LangGraph's `interrupt()` pattern to prevent unverified autonomous actions.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
