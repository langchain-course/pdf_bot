from dotenv import load_dotenv

from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import chainlit as cl
from chainlit.types import AskFileResponse

load_dotenv()


@cl.on_chat_start
async def start():
    files = None
    # 等待用户上传文件
    while files is None:
        files = await cl.AskFileMessage(
            content='请上传你要提问的PDF文件',
            # 这里只运行 PDF 的文件
            accept=["application/pdf"]
        ).send()

    _file = files[0]

    # 文件还没加载成功之前显示一个消息提示
    msg = cl.Message(content=f'正在处理处理: `{_file.name}`...')
    await msg.send()

    # 将上传的文件保存到服务器本地
    file_path = (f'./tmp/{_file.name}')
    open(file_path, 'wb').write(_file.content)

    # 加载 PDF 文档
    docs = PyMuPDFLoader(file_path).load()

    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(docs)

    # 创建 Chroma 存储
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_documents)(
        split_docs, embeddings, collection_name=_file.name
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        output_key='answer',
        return_messages=True,
    )

    # 基于 Chroma 存储创建一个问答链
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(
            temperature=0,
            model='gpt-4-1106-preview',
        ),
        chain_type='stuff',
        retriever=docsearch.as_retriever(),
        memory=memory,
    )

    msg.content = f'`{_file.name}` 处理完成，请开始你的问答。'
    await msg.update()

# @cl.on_message
# async def process_response(res):
#     answer = res["answer"]
#     await cl.Message(content=answer).send()
