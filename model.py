from semantic_router.layer import RouteLayer
from semantic_router import Route

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
# import chainlit as cl
from langchain.prompts.chat import(
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain.chains import LLMChain
from langchain.evaluation import load_evaluator
from langchain.evaluation import EvaluatorType
import chainlit as cl

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    #context_recall,
    context_precision,
)

import os
import openai
 
openai.api_key = os.environ["OPENAI_API_KEY"]
# import os

# we could use this as a guide for our chatbot to avoid political conversations
AssetSense = Route(
    name="AssetSense",
    utterances=[
        "AssetSense is a modern asset performance monitoring platform",
        "Transforms Power Generation",
        "Features of AssetSense",
        "Benefits of AssetSense",
        "Asset Monitoring Technologies Don’t Work",
        "elevates both plant and operator performance",
        "Asset Performance Monitoring Solutions"
        "Power Generation Leaders Are Saying"
        "Operator rounds"
        # "performance analytics"
        # # "predictive maintenance"
        # "asset maintenance is important"
        # "success stories of company"
        # "contact information details"
        # "roles overview"
        # "executive"
        # "performance manager"
        # "plant leadership"
        # "it and cio"
        # "solution overview"
        # "digital transformation"
        # "industrial iot"
        # "cloud computing"
        # "industrial iot"
        # "modern analytics"
        # "mobile apps"
        # "integration"
        # "security"
        # "white paper"
        # "product demo"
        # "company profile"
        # "contact us"
        # "privacy policy"
        ""
],
)

# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt


# we place both of our decisions together into single list
routes = [AssetSense]

import os
from semantic_router.encoders import CohereEncoder, OpenAIEncoder

# for Cohere
os.environ["COHERE_API_KEY"] = ""
encoder = CohereEncoder()

# # or for OpenAI
# os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
# encoder = OpenAIEncoder()
r1 = RouteLayer(encoder=encoder, routes=routes)
print("routing completed")
DB_FAISS_PATH = 'demovs3'
template = """You are ceo of the company. Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=template,
                            input_variables=['context', 'question'])
    return prompt

def get_retreiver():
    return retreiver

def retrieval_qa_chain(llm, prompt, db):
    retriever=db.as_retriever( search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k": 4} )
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# from huggingface_hub import hf_hub_download
model_name_or_path = "TheBloke/Llama-2-7B-chat-GGUF"
model_basename = "llama-2-7b-chat.Q4_K_M.gguf"
MODEL_PATH ="/Users/developer/Documents/srija/llama-2-7b-chat.Q4_K_M.gguf"

def load_llm():

    n_gpu_layers = 50
    n_batch = 512
    llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx = 4096,
    max_new_tokens=4096
)
    cl.user_session.set("llm", llm)
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'mps'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k": 4} )
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa,retriever



def final_result(query):
    qa_result,retriever = qa_bot()
    llm = load_llm()

    evaluator=load_evaluator(EvaluatorType.CRITERIA,criteria='conciseness', llm = llm)
    response = qa_result.invoke({'query': query})
    eval_result=evaluator.evaluate_strings(
        prediction=response,
        input=query
    )
    print("VALUE:",eval_result['value'])
    print("SCORE:",eval_result['score'])
    print("REASON:\n",eval_result['reasoning'])
    return response

# query="What is Assetsense?"
# if(r1(query).name=='AssetSense'):
#  final_result(query)
# else:
#  print("No information")

@cl.on_chat_start
async def start():
    chain,retriever = qa_bot()
    questions = ["What is AssetSense?",
            "Why should customers choose AssetSense",
            ]
    print("questons")
    print("questions")
    # ground_truths = [["The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."],
                    # ["The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."],
    #                 ["The president asked Congress to pass proven measures to reduce gun violence."]]
    answers = []
    contexts = []
    
    # Inference
    for query in questions:
      answers.append(chain.invoke(query)['result'])
      ret_docs = retriever.get_relevant_documents(query)
      contexts.append([docs.page_content for docs in ret_docs])
    print("answers contexts created")
    
    # To dict
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
       # "ground_truths": ground_truths
    }
    print("Data created")
    
    # Convert dict to dataset
    dataset = Dataset.from_dict(data)
    print("dataset created")
    print("dataset created")
    print("dataset created")
    
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        #context_recall,
        context_precision,
    )
    
    result = evaluate(
        dataset = dataset,
        metrics=[
            context_precision,
           # context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )
    
    df = result.to_pandas()
    print(result)
    print(df)
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to  AssetBot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    if(r1(message.content).name=='AssetSense'):
        print("Hello")
        res = await chain.ainvoke(message.content, callbacks=[cb])
        print(res)
        evaluator=load_evaluator(EvaluatorType.CRITERIA,criteria='relevance', llm = cl.user_session.get("llm"))
    
        eval_result=evaluator.evaluate_strings(
            prediction=res,
            input=message.content
        )
        print("VALUE:",eval_result['value'])
        print("SCORE:",eval_result['score'])
        print("REASON:\n",eval_result['reasoning'])
        answer = res["result"]
        sources = res["source_documents"]
        print(sources)
    else:
        answer = "No information"

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += "\nNo sources found"

    await cl.Message(content=answer).send()
    # print(response)
    # print("Time: ",end-start)
# final_result("what is assetsense")
# load_llm()
