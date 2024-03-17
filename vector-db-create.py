#vector-db-create.py
# create a vector database from a pdf file
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


#Step1.Load PDF text
loaders = [PyPDFLoader('./pdf/公司考勤与休假.pdf')]  #list type

docs = []
for file in loaders:
    docs.extend(file.load())
#print("docs type:",type(docs))    # 返回一个可迭代的list，Iterable[Document]，即：[Document(page_content='xxxx', metadata={'source':'./pdf/公司考勤与休假.pdf', 'page': 4})]


#Step2.Split text to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,length_function=len)
chunks = text_splitter.split_documents(docs) #因为docs不是纯文本string，是list，所以用split_documents;否则可用split_text！

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})
#print(len(docs))

#save to disk at location:"./chroma_db_test1"
vectorstore = Chroma.from_documents(chunks, embedding_function, persist_directory="./chroma_db_test1")
print(vectorstore._collection.count())
