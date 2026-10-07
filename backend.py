import os
import sqlite3
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage
from typing import TypedDict,Annotated,Any,Optional,Dict
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import tools_condition,ToolNode
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.types import interrupt,Command
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
#==============================Setup======================================================
load_dotenv()
model=AzureChatOpenAI(deployment_name='gpt-4.1-mini')
os.environ['LANGCHAIN_PROJECT']='Chatbot_project'
conn=sqlite3.connect(database='chatbot',check_same_thread=False)
embeddings=AzureOpenAIEmbeddings(model='text-embedding-3-small')

_THREAD_RETRIEVERS:Dict[str,Any]={}
_THREAD_METADATA:Dict[str,dict]={}

#=============================Chat Title Setup==========================================
conn.execute("CREATE TABLE IF NOT EXISTS chat_titles (thread_id TEXT PRIMARY KEY, title TEXT)")
conn.commit()

def generate_and_save_title(thread_id: str, first_message: str):
    """Synchronously asks the LLM for a title and saves it to SQLite."""
    prompt = f"Summarize this in 3-4 words for a chat title. No quotes. Query: {first_message}"
    # Standard synchronous call
    response = model.invoke([SystemMessage(content="You are a title generator."), HumanMessage(content=prompt)])
    title = response.content.strip().replace('"', '')
    
    conn.execute("INSERT OR IGNORE INTO chat_titles (thread_id, title) VALUES (?, ?)", (thread_id, title))
    conn.commit()

def get_thread_title(thread_id: str):
    """Fetches the title, defaults to truncated thread_id if none exists."""
    cursor = conn.execute("SELECT title FROM chat_titles WHERE thread_id = ?", (thread_id,))
    row = cursor.fetchone()
    return row[0] if row else str(thread_id)[:8] + "..."

#==============================DOCUMENT UPLOADING SETUP=================================
def thread_retriever(thread_id:Optional['str']):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS.get(str(thread_id))

def pdf_ingestion(file_bytes:bytes,file_name:Optional[str],thread_id:str):
    if not file_bytes:
        raise ValueError('No bytes recieved for ingestion')

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        temp_file.write(file_bytes)
        temp_file.flush()
        temp_path = temp_file.name
    finally:
        temp_file.close()

    try:
        loader=PyMuPDFLoader(temp_path)
        docs=loader.load()

        splitters=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,separators=['\n\n','\n',' ',''])

        chunks=splitters.split_documents(docs)

        vector_store=FAISS.from_documents(chunks,embeddings)
        retriever= vector_store.as_retriever(search_type='similarity',search_kwargs={'k':4})

        t_key=str(thread_id)
        _THREAD_RETRIEVERS[t_key]=retriever
        _THREAD_METADATA[t_key]={
            'file_name':file_name or os.path.basename(temp_path),
            'documents':len(docs),
            'chunks':len(chunks)
        }

        return _THREAD_METADATA[t_key]

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def thread_has_document(thread_id:str):
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_metadata(thread_id):
    return _THREAD_METADATA.get(str(thread_id),{})

#===============================State Setup=============================================

class chatbot_state(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

#===============================Tools===================================================

@tool
def rag_tool(query:str,thread_id:Optional[str]=None)->Dict:
    '''Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.'''

    retriever=thread_retriever(thread_id)
    if retriever is None:
        return {'error':'No document uploaded please upload a document first',
                'query':query}

    result=retriever.invoke(query)
    context=[doc.page_content for doc in result]
    metadata=[doc.metadata for doc in result]

    return {
        'query':query,
        'metadata':metadata,
        'context':context,
        'source_file':_THREAD_METADATA.get(str(thread_id),{}).get('file_name')
    }


@tool
def stock_price(symbol:str):
    '''checks stock price at realtime'''
    api=os.getenv('INFOWAY_API_KEY')
    url=f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol.upper()}&apikey={api}'
    r= requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol:str,quantity:int)->dict:
    """Call this tool immediately whenever the user wants to buy or purchase shares of a stock.
    Do not ask for confirmation in chat; invoke this tool directly."""

    decision=interrupt(f'Approve buying {quantity} shares of {symbol}.')

    if isinstance(decision,str) and decision.lower()=='yes':
        return{
            'status':'success',
            'message':f'purchase order placed for {quantity} shares of {symbol}.',
            'symbol':symbol,
            'quantity':quantity
        }
    else:
        return{
                    'status':'cancelled',
                    'message':f'purchase order  for {quantity} shares of {symbol} declined by the user.',
                    'symbol':symbol,
                    'quantity':quantity
                }

search_tool=DuckDuckGoSearchRun()

tools=[stock_price,search_tool,purchase_stock,rag_tool]
tool_node=ToolNode(tools)
llm_tool=model.bind_tools(tools)

#=============================Chatbot function============================================
def chatbot(state:chatbot_state,config=None):
    thread_id=None
    if config and isinstance(config,dict):
        thread_id=config.get('configurable',{}).get('thread_id')

    system_message=SystemMessage(content=("You are a helpful assistant. For questions about the uploaded PDF for this chat, "
            f"call the `rag_tool` and include the thread_id `{thread_id}`. "
            "You can also use web search, stock price lookups, and purchase tools when requested. "
            "If no document is available, ask the user to upload a PDF in the sidebar."))

    messages=[system_message,*state['messages']]
    response=llm_tool.invoke(messages,config=config)

    return {'messages':[response]}

#============================Graph==========================================================

graph=StateGraph(chatbot_state)
checkpointer=SqliteSaver(conn=conn)
graph.add_node('chatbot',chatbot)
graph.add_node('tools',tool_node)
graph.add_edge(START,'chatbot')
graph.add_conditional_edges('chatbot',tools_condition)
graph.add_edge('tools','chatbot')
comp=graph.compile(checkpointer=checkpointer)

#=============================Thread Loader Function=========================================
def thread_loader():
    unique_threads=set()
    for checkpoints in checkpointer.list(None):
        threads = checkpoints.config['configurable'].get('thread_id')
        if threads:
            unique_threads.add(threads)
    return [{'thread_id':tid} for tid in unique_threads]
