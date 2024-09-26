#Import libraries
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
import tempfile

#Configure streamlit app
st.set_page_config(page_title="Family Medicine Clinical Practice Guidelines", page_icon="➳❥", layout = "centered")
st.title("📖 Family Medicine Clinical Practice Guidelines \n Information derived from published clinical practice guideliness")

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

# Configure cached vector database
@st.cache_resource
def config_vector_db():
    s3_client = boto3.client('s3')
    bedrock_client = boto3.client('bedrock-runtime')
    bedrock_embeddings = BedrockEmbeddings(client=bedrock_client)
    all_documents = []

    # Specify bucket_name and folder_path based on your S3 bucket and folder names.
    bucket_name = '<S3 bucket name>' # S3 bucket name
    folder_path = '<S3 folder name in specified bucket>' # S3 folder name in specified bucket

    # List all files in the S3 folder
    s3_objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)

    # Iterate over all files in the S3 folder
    for obj in s3_objects.get('Contents', []):
        if obj['Key'].endswith(".pdf"):  # Check if the file is a PDF
            pdf_file_key = obj['Key']
            file_name = os.path.basename(pdf_file_key)

            # Download the PDF file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_pdf_file:
                s3_client.download_fileobj(bucket_name, pdf_file_key, temp_pdf_file)
                temp_pdf_file_path = temp_pdf_file.name

            print(f"Processing {file_name}...")

            # Load and process the PDF file
            loader = PyPDFLoader(temp_pdf_file_path)
            pages = loader.load_and_split()

            # Add the pages to the list of all documents
            all_documents.extend(pages)

            print(f"Finished processing {file_name}")

            # Clean up the temporary file
            os.remove(temp_pdf_file_path)

    # Creating the FAISS vectorstore from the documents
    vectorstore_faiss = FAISS.from_documents(all_documents, bedrock_embeddings)
    return vectorstore_faiss

#Configuring the llm and vector store
llm = config_llm()
vectorstore_faiss = config_vector_db()

context = """
You are a knowledgeable assistant specialized in various clinical practice guidelines 
exclusively for licensed healthcare providers. Your answers should be derived from the main source of reference for the particular
subject. For example, if you are asked about blood pressure or hypertension, you must refer to the hypertension guideliness instead of
the other guideliness although they may contain information about hypertension too. Your responses should be concise, 
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
