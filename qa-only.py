from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate 

#"ConversationalRetrievalChain" handles the chat history and the retrieval of similar document chunks to be added to the prompt.
from langchain.chains import ConversationalRetrievalChain 

#虽然langchain_openai模块主要是为了与OpenAI API交互而设计的，
#但是它的设计使得它可以与任何遵循类似API的服务进行交互。
#这意味着，如果有其他服务提供了与OpenAI API类似的API，ChatOpenAI类也可以用来与这些服务交互。
#这个项目就是用的与openAI API类似的API！
from langchain_openai import ChatOpenAI 

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding_function=HuggingFaceEmbeddings(model_name=embedding_model)

#Load vector database from the disk at location:"./chroma_db_test1"
vector_db = Chroma(persist_directory="./chroma_db_test1", embedding_function=embedding_function) #加载之前创建的知识库

question = "我2013年入职的，我今年休假3天了，还有几天年假?"
print("\n查找知识库相似知识点:", question)


'''
#k=2表示返回最相似的前两个知识点,
#虽然代码没有直接显示将问题转化为向量的步骤，但这个过程是在similarity_search方法内部隐式地进行的！
#根据问题，去向量数据库检索向量距离最近的相关的内容！
search_results = vector_db.similarity_search(question, k=2) 
search_results_string = ""
for result in search_results:
    search_results_string += result.page_content + "\n\n"
print("search_result_sting:",search_results_string)
'''

#temperature：控制了模型生成文本的随机性。temperature值越低，生成的文本越确定，即模型的输出更可能是固定的
#base_url：这个参数指定了API请求的基础URL。在这个例子中，你指定一个本地服务器的地址，这意味着你可能正在使用一个本地运行的服务来模拟或代理OpenAI API的调用。这对于在没有直接访问OpenAI API的情况下进行开发和测试非常有用。
llm = ChatOpenAI(temperature=0.5, base_url="http://localhost:8888/v1", api_key="not-needed") 


# Build prompt
template = """
    使用你的知识，并结合以下的内容回答问题。 \
    如果你不知道答案，就说你不知道，不要试图编造答案。 \
    最多使用三句话，尽可能简明扼要地回答问题。回答开头可以说"感谢提问！" \
    内容：{context} \
    问题: {question}
    答案:
    """
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)



#使用ConversationBufferMemory类创建了一个对话历史记忆memory。这个记忆将用于存储和管理对话的历史记录。
memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer", return_messages=True)
# Run chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

print("\n\nRunning AI\n\n")

result = qa_chain.invoke({"question": question})
print(result["answer"])
