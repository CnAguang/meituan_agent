import os
from typing import Optional, Dict, Any

import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.embeddings import ZhipuAIEmbeddings,BaichuanTextEmbeddings
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

os.environ['USER_AGENT'] = 'Your-App-Name/1.0'
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = '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'

#加载ollama模型
model = ChatOpenAI(model='meitua1.5bmodel:latest',base_url='http://localhost:11434/v1',openai_api_key='123',streaming=False)
model.invoke('美团外卖支持哪些支付方式')

#加载数据
loader = TextLoader(
    file_path=r'C:\Users\15425\Desktop\常见问题文档.txt',
    encoding='utf-8',
)


documents = loader.load()

#进行分割
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
splits = splitter.split_documents(documents)
#分割后的文本进行存储
embedding_store = Chroma.from_documents(documents=splits,embedding=ZhipuAIEmbeddings(api_key='a0405e581a1745b1a1d563ef386ff02f.0Q82NTjEmMuElQV3'))
#变为检索器
retriever = embedding_store.as_retriever()

#整合模型
#创建prompt
system_prompt = """you are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.Use three sentences
maximum and keep the answer concise. \n
{context}"""
prompt = ChatPromptTemplate.from_messages(
    [
        ('system',system_prompt),
        MessagesPlaceholder('chat_history'),
        ('user','{input}')
    ]
)

chain1 = create_stuff_documents_chain(model,prompt)

#如果继续问”它有什么主要技术“，模型理解不了它是谁，就算有了历史记录，没有检索也回答不了，所以需要历史记录+检索
contextualize_q_system_promote = """
Given a chain history and the latest user question which might reference context in the chat history,
formulate a standalone question which can be understood without the chat history. DO NOT answer the question,
just reformulate it if needed and otherwise return it as is.
"""
retriever_history_temp = ChatPromptTemplate.from_messages([
    ('system',contextualize_q_system_promote),
    MessagesPlaceholder('chat_history'),
    ('user','{input}')
])

history_chain = create_history_aware_retriever(model,retriever,retriever_history_temp)

store = {}
from langchain_community.chat_message_histories import ChatMessageHistory
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain2 = create_retrieval_chain(history_chain,chain1)
result_chain = RunnableWithMessageHistory(
    chain2,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    answer_messages_key='answer',
)

# result_1 = result_chain.invoke(
#     {'input': '你是谁'},
#     config = {'configurable':{'session_id':'abc123'}}
# )
# print(result_1)
from fastapi import FastAPI, HTTPException
from langserve import add_routes
# 定义输入模型
# 新增模型定义
class ChatInput(BaseModel):
    input: str
    session_id: str = "default"
    # configurable: Optional[Dict[str, Any]] = None

class ChatOutput(BaseModel):
    answer: str

app = FastAPI(title='美团小帮手',version='1.0',description='微调后做了rag检索还带历史记录的一个助手')
#跨域
from fastapi.middleware.cors import CORSMiddleware  # 新增导入
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 允许的前端地址
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法（GET/POST等）
    allow_headers=["*"],  # 允许所有请求头
)
# 自定义端点处理带历史的请求
@app.post("/chat")
async def chat_endpoint(data: ChatInput):
    try:
        result = result_chain.invoke(
            {"input": data.input},
            config={"configurable": {"session_id": data.session_id}}
        )
        # 处理可能的BaseMessage对象
        answer = result["answer"]
        if isinstance(answer, BaseMessage):
            answer = answer.content
        return {"answer": answer,'code':200}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

add_routes(
    app,
    result_chain,
    path='/langchain_meituan',
    input_type=ChatInput,
    output_type=ChatOutput,
    config_keys=['configurable']
)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=8000)