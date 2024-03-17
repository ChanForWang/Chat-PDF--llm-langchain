import streamlit as st
from htmlTemplates import css,bot_template,user_template
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#Embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings
#vector database
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI  #类似openAI的方式调用llm
from langchain.prompts import PromptTemplate 
from langchain.memory import ConversationBufferMemory
#"ConversationalRetrievalChain" handles the chat history（所以import ConversationBufferMemory） 
# and the retrieval of similar document chunks to be added to the prompt.
from langchain.chains import ConversationalRetrievalChain 
from langchain_community.chat_models import ChatZhipuAI
#from zhipuai import ZhipuAI
from dotenv import load_dotenv
import os

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_db(chunks):
    #save：将pdf的chunks变成嵌入向量，嵌入到向量数据库中，并且本地保存！
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})
    vector_db = Chroma.from_texts(chunks, embedding)
    return vector_db

def get_conversation_chain(vector_db,ZhiPuAI_API_KEY):    
    #llm = ZhipuAI(api_key=ZhiPuAI_API_KEY,base_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",temperature=0.2) 
    llm = ChatZhipuAI(api_key=ZhiPuAI_API_KEY,model="chatglm_turbo",temperature=0.3)
    #temperature：控制了模型生成文本的随机性。temperature值越低，生成的文本越确定，即模型的输出更可能是固定的
    #base_url：这个参数指定了API请求的基础URL。在这个例子中，你指定一个本地服务器的地址，这意味着你可能正在使用一个本地运行的服务来模拟或代理OpenAI API的调用。这对于在没有直接访问OpenAI API的情况下进行开发和测试非常有用。
    #llm = ChatOpenAI(temperature=0.2, base_url="http://localhost:8888/v1", api_key="not-needed") 
    
    # Build prompt which insert into llms
    template = """
    使用你的知识，并结合以下的内容回答问题。 \
    如果你不知道答案，就说你不知道，不要试图编造答案。 \
    最多使用三句话，尽可能简明扼要地回答问题。回答开头可以说"感谢提问！" \
    内容：{context} \
    问题: {question} \
    答案:
    """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    #创建了一个对话历史记忆memory。这个记忆将用于存储和管理对话的历史记录。
    memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer", return_messages=True)
    #ConversationalRetrievalChain" handles the 1.chat history and the 2.retrieval of similar document chunks to be added to the prompt.
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(), #Create a retriever instance of the vector store
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return conversation_chain

def handle_userinput(user_question):
    #这行代码调用了存储在st.session_state.conversation中的对话链实例（即：get_conversation_chain(vector_db)），并将用户的问题作为参数传递给它
    response = st.session_state.conversation({'question': user_question})
    #st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) #获取用户内容
        else:
            message.content =message.content.replace("\\n","<br>")
            st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True) #获取AI回答的内容




def main():
    load_dotenv()
    if os.getenv("ZhiPuAI_API_KEY"):
        ZhiPuAI_API_KEY=os.getenv("ZhiPuAI_API_KEY")
    else:
        with st.sidebar:
            ZhiPuAI_API_KEY=st.text_input("Input your ZhipuAI API Key!")

    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(bot_template.replace("{{MSG}}", "Hi,What could I help you?"), unsafe_allow_html=True)


    #这段代码的目的是在Streamlit应用程序的会话状态（session state）中初始化两个键值对："conversation"和"chat_history"。这两个键值对用于存储对话链实例和对话历史记录。

    #检查"conversation"键：首先，代码检查st.session_state字典中是否存在"conversation"键。
    #如果不存在，那么就在字典中添加一个键值对，键为"conversation"，值为None。
    #这意味着在用户上传PDF文件并点击“Process”按钮之前，对话链实例（conversation chain）不会被创建，
    #因为它的值被初始化为None。
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    #检查"chat_history"键：类似地，代码检查st.session_state字典中是否存在"chat_history"键。
    #如果不存在，那么就在字典中添加一个键值对，键为"chat_history"，值为None。
    #这意味着在用户开始与聊天机器人交互之前，对话历史记录（chat history）不会被创建，因为它的值被初始化为None。
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        try:
            handle_userinput(user_question)
        except TypeError as e:
            #st.error(str(e))
            st.warning("Please upload your PDFs first!",icon="⚠️")
            
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload/Drag your PDFs here.'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                #1.Get pdf text
                raw_text = get_pdf_text(pdf_docs)
             
                #2.Get the text chunks
                text_chunks = get_text_chunks(raw_text) #return a list
                #st.write(text_chunks)

                #3.Create vector store
                vector_db = get_vector_db(text_chunks)
                #st.write(vector_db._collection.count())

                # 4.Create conversation chain，并用 st.session_state.conversation来存储对话链实例，
                #这样在用户提交问题时，应用程序可以使用这个对话链来处理问题并生成答案
                #当用户点击“Process”按钮并上传PDF文件后，"conversation"键的值将被【更新】为一个新创建的对话链实例
                st.session_state.conversation = get_conversation_chain(vector_db,ZhiPuAI_API_KEY)

        
            

if __name__ == "__main__":
    main()