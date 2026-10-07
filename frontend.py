from backend import comp, thread_loader, pdf_ingestion, thread_metadata, generate_and_save_title, get_thread_title
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import uuid
import streamlit as st
from langgraph.types import Command

#=======================================UTILITY FUNCTIONS====================================================

def new_thread():
    thread_id=uuid.uuid4()
    return str(thread_id)

def new_chat():
    thread_id=new_thread()
    st.session_state['thread_id']=thread_id
    st.session_state['thread_list'].append({'thread_id':thread_id})
    st.session_state['message_history']=[]
    st.rerun()

def thread_message(thread_id):
    messages=comp.get_state(config={'configurable':{'thread_id':thread_id}})
    thread_messages=messages.values.get('messages',[])
    return thread_messages


#======================================Initialisation===========================================================

if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

if 'ingested_docs' not in st.session_state:
    st.session_state['ingested_docs']={}

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=str(new_thread())

if 'thread_list' not in st.session_state:
    saved_threads=thread_loader()
    existing_threads={t['thread_id'] for t in saved_threads}
    if st.session_state['thread_id'] not in existing_threads:
        saved_threads.append({'thread_id':st.session_state['thread_id']})
    st.session_state['thread_list']=saved_threads

for messages in st.session_state['message_history']:
    with st.chat_message(messages['role']):
        st.text(messages['content'])


#=====================================SideBar================================================================

st.sidebar.title('GPT')

if st.sidebar.button('New Chat'):
    new_chat()

thread_key=str(st.session_state['thread_id'])
thread_docs=st.session_state['ingested_docs'].setdefault(thread_key,{})

if thread_docs:
    latest_doc=list(thread_docs.values())[-1]
    st.sidebar.success(f"Using : {latest_doc.get('file_name')}")
else:
    st.sidebar.info('No PDF uploaded')

uploaded_pdf=st.sidebar.file_uploader('Upload a PDF for this chat',type=['pdf'])

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f'{uploaded_pdf.name} already processed for this chat')

    else:
        with st.sidebar.status('Indexing PDF',expanded=True) as status_box:
            summary= pdf_ingestion(uploaded_pdf.getvalue(),thread_id=thread_key,file_name=uploaded_pdf.name)
            thread_docs[uploaded_pdf.name]=summary
            status_box.update(label='✅ PDF indexed',state='complete',expanded=False)
            st.rerun()

st.sidebar.markdown('---')
st.sidebar.subheader('Past conversation')


for threads in st.session_state['thread_list'][::-1]:
    title=get_thread_title(threads['thread_id'])
    if st.sidebar.button(title,key=f"btn_{threads['thread_id']}"):
        messages=thread_message(threads['thread_id'])
        temp_msg=[]
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role='user'
                temp_msg.append({'role':role,'content':msg.content})
            elif isinstance(msg,AIMessage):
                role='assistant'
                temp_msg.append({'role':role,'content':msg.content})
        st.session_state['message_history']=temp_msg
        st.session_state['thread_id']=threads['thread_id']
        st.rerun()



# ======================================= Chat Area ================================================================

user_input = st.chat_input('Type Here')

if user_input:
    if len(st.session_state['message_history'])==0:
        generate_and_save_title(st.session_state['thread_id'], user_input)

    current_thread_id = str(st.session_state['thread_id'])
    config={'configurable':{'thread_id':st.session_state['thread_id']},'metadata':{'thread_id':st.session_state['thread_id']},'run_name':'chat_run'}
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.text(user_input)

    # 1. Check if this thread is currently paused on an interrupt
    state = comp.get_state(config)
    is_paused = bool(state.tasks and len(state.tasks[0].interrupts) > 0)

    # 2. If paused, resume with user's text ("yes"/"no"); otherwise, pass as a new prompt
    if is_paused:
        graph_input = Command(resume=user_input)
    else:
        graph_input = {'messages': [HumanMessage(content=user_input)]}

    # 3. Stream the response
    with st.chat_message('assistant'):
        response = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in comp.stream(
                graph_input,
                config=config,
                stream_mode='messages'
            )
            if metadata.get('langgraph_node') == 'chatbot' and message_chunk.content
        )

        # 4. If the graph just paused on an interrupt during this turn, show the prompt
        new_state = comp.get_state(config)
        if new_state.tasks and len(new_state.tasks[0].interrupts) > 0:
            response = new_state.tasks[0].interrupts[0].value
            st.text(response)

    if response:
        st.session_state['message_history'].append({'role': 'assistant', 'content': response})