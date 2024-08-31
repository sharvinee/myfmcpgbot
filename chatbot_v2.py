#Import libraries
import os
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockLLM
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


#Configure streamlit app
st.set_page_config(page_title="Type 2 Diabetes Mellitus Clinical Practice Guidelines", page_icon="➳❥", layout = "centered")
st.title("📖 T2DM Clinical Practice Guidelines \n - VA/DoD Clinical Practice Guideline for the Management of Type 2 Diabetes Mellitus")

#Define convenience functions
@st.cache_resource
def config_llm():
    myclient = boto3.client('bedrock-runtime')

    mymodel_kwargs = { 
        "max_tokens_to_sample": 512,
        "temperature":0.1,  
        "top_p":1
    }  

    mymodel_id = "anthropic.claude-instant-v1"

    llm = BedrockLLM(model_id=mymodel_id, client=myclient, model_kwargs=mymodel_kwargs)
    return llm

@st.cache_resource
def config_vector_db(pdf_file_path):
    client = boto3.client('bedrock-runtime')
    bedrock_embeddings = BedrockEmbeddings(client=client)
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load_and_split()
    
    #Creating the FAISS vector store from the documents
    vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)
    return vectorstore_faiss

my_pdf_file_path = "./documents/file.pdf"

#Configuring the llm and vector store
llm = config_llm()
vectorstore_faiss = config_vector_db(my_pdf_file_path)

context = """
You are a knowledgeable assistant specialized in Type 2 Diabetes Mellitus clinical practice guidelines 
exclusively for licensed healthcare providers. Your responses should be concise, 
and aligned with the clinical practice guidelines provided to you. Your role is to support doctors by providing 
relevant information and guidance, not to offer medical advice to patients or non-medical persons.

Always include the following disclaimer at the end of each response: 

"This information is intended for use by licensed healthcare providers only and is not a substitute 
for professional medical advice, diagnosis, or treatment. This tool is currently under development, 
and while we strive for accuracy, there may be limitations in the information provided." 

Here is the question from a user:"""

#Creating the template   
my_template = """

Human: 

{chat_history}

<Information>
{info}
</Information>

{context}

{input}

Assistant:
"""

#Configure prompt template
prompt_template = PromptTemplate(
    input_variables=['context', 'info', 'input', 'chat_history'],
    template= my_template
)

#Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory = msgs, memory_key = 'chat_history', return_messages = True)
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello, Dr! I am your CPG Bot. Please ask me your doubts.")

question_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = vectorstore_faiss.as_retriever(),
    combine_docs_chain_kwargs= {"prompt" : prompt_template},
    memory = memory,
    condense_question_prompt= CONDENSE_QUESTION_PROMPT
)

question_chain = prompt_template | llm 

#Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

#If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    #Retrieve relevant document using a similarity search
    docs = vectorstore_faiss.similarity_search_with_score(prompt)
    info = ""
    for doc in docs:
        info += doc[0].page_content + '\n'

    #invoke llm
    output = question_chain.invoke({"context": context, "input": prompt, "info": info, "chat_history": memory.load_memory_variables({})})
    
    #adding messages to history
    msgs.add_user_message(prompt)
    msgs.add_ai_message(output)

    #display the output

    st.chat_message("ai").write(output)