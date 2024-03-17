# MultiPDF Chat App
Base on ChatGLM+Chromadb+Langchain+Streamlit to build a PDF chatbot!
(Hopefully, word,excel,markdown,txt...will be introduced.Stay tuned!)


## Introduction
------------
The MultiPDF Chat App is a Python application that allows you to chat with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.

## Give a try
------------
[Click Click ME](https://chat-pdf-glm.streamlit.app/) to have a try on this real-time PDF-Chatbot. 

## How It Works
------------

![MultiPDF Chat App Diagram](./doc/PDF-LangChain.jpg)

The application follows these steps to provide responses to your questions:

1. PDF Loading: The app reads multiple PDF documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

## Dependencies and Installation
----------------------------
To install the MultiPDF Chat App, please follow these steps:

1. Clone the repository to your local machine.
   ```
   git clone https://github.com/ChanForWang/Chat-PDF--llm-langchain.git
   ```

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from [ChatGLM](https://open.bigmodel.cn/usercenter/apikeys) and add it to the `.env` file in the project directory.
```commandline
ZhiPuAI_API_KEY=your_secrit_api_key
```

## Usage
-----
To use the MultiPDF Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the ChatGLM API key to the `.env` file.

2. Run the `app.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded PDFs using the chat interface.

## Contributing
------------
This repository is intended for educational purposes and accept any contribution. Feel free to utilize and enhance the app based on your own requirements.

## License
-------
The MultiPDF Chat App is released under the [Apache-2.0 license](http://www.apache.org/licenses/).



# 多媒体PDF聊天应用。

基于ChatGLM+Chromadb+Langchain+Streamlight构建一个PDF聊天机器人！(希望能介绍Word、Excel、Markdown、txt……，敬请关注！)。


## 简介
-------
MultiPDF聊天应用程序是一款允许您与多个PDF文档聊天的Python应用程序。您可以使用自然语言询问有关PDF的问题，应用程序将根据文档内容提供相关回答。这款应用程序使用语言模型来生成对您的查询的准确答案。请注意，该应用程序只会回答与加载的PDF相关的问题。

## 浅尝一下
------------
[点我快点我](https://chat-pdf-glm.streamlit.app/)试试此实时PDF-chatbot！

## 工作原理
-------
![MultiPDF聊天应用图](./doc/pdf-LangChain.jpg)

应用程序遵循以下步骤为您的问题提供答复：

1.PDF加载：该应用程序读取多个PDF文档，并提取其文本内容。

2.文本组块：将提取的文本分成可以有效处理的小块。

3.语言模型：应用程序利用语言模型来生成文本块的矢量表示(嵌入)。

4.相似度匹配：当你问一个问题时，应用程序会将它与文本块进行比较，并找出语义上最相似的。

5.响应生成：将选择的组块传递给语言模型，语言模型根据PDF的相关内容生成响应。

## 依赖和安装
-------
要安装MultiPDF聊天应用程序，请执行以下步骤：

1.将存储库克隆到您的本地计算机。
```
Git克隆https://github.com/ChanForWang/Chat-PDF--llm-langchain.git。
```

2.通过运行以下命令安装所需的依赖项：
```
pip install -r requirements.txt。
```

3.从[ChatGLM](https://open.bigmodel.cn/usercenter/apikeys)获取接口密钥，添加到工程目录的`.env`文件中。
```commandline
ZhiPuAI_API_KEY=您的SECRIT_API_KEY。
```

## 使用方法
-------
要使用MultiPDF聊天应用程序，请执行以下步骤：

1.确保安装了所需的依赖项，并将ChatGLM API密钥添加到`.env`文件中。

2.使用Streamlight命令行工具运行`app.py`文件。执行以下命令：
```
Streamlight Run app.py。
```

3.应用程序将在您的默认Web浏览器中启动，显示用户界面。

4.按照提供的说明将多个PDF文档加载到应用程序中。

5.使用聊天界面用自然语言询问有关加载的PDF的问题。

## 贡献力量
-------
此存储库旨在用于教育目的，并接受任何贡献。您可以根据自己的需求随意使用和增强该应用程序。

## 许可证
-------
多媒体PDF聊天应用程序是在[APACHE-2.0许可](http://www.apache.org/licenses/)下发布的