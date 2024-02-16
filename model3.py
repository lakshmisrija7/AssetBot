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
    context_precision,

)
from ragas.metrics.critique import harmfulness
from ragas.langchain.evalchain import RagasEvaluatorChain 

import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
import os

# we could use this as a guide for our chatbot to avoid political conversations
AssetSense = Route(
    name="AssetSense",
    utterances=[
        '''AssetSense is a modern asset performance monitoring platform''',
        '''Transforms Power Generation''',
        '''Features of AssetSense''',
        '''Benefits of AssetSense''',
        '''Asset Monitoring Technologies Don’t Work''',
        '''elevates both plant and operator performance''',
        '''Asset Performance Monitoring Solutions'''
        '''Power Generation Leaders Are Saying'''
        '''Why is there an increased pressure on power plants to transform their operations?'''
        '''What is the potential cost of an asset failure in a power plant?'''
        '''What is the main challenge faced by power plants in terms of equipment upgrades, maintenance, or replacement?'''
        '''Why might traditional asset monitoring technologies fall short in power generation?'''
        '''In what ways does AssetSense cater to different types of power generation, such as fossil fuel, hydro, wind, and biofuel?'''
        ''' AssetSense assist in viewing and managing asset dataHow does AssetSense address the limitations of other asset monitoring technologies?'''
        '''AssetSense support fleet-wide monitoring and decision-making'''
        '''How does AssetSense address the limitations of other asset monitoring technologies?'''
        '''interested parties request a demo or get in touch with AssetSense'''
        '''address the complexities and challenges faced by traditional asset monitoring systems'''
        ''' digital transformation efforts in power plants fail, address the complexities and challenges faced by traditional asset monitoring systems'''
        '''challenges are highlighted when power plants choose solutions for digital transformationaddress the complexities and challenges faced by traditional asset monitoring systems'''
        '''costly and lengthy implementations that come with most technology rolloutsaddress the complexities and challenges faced by traditional asset monitoring systems'''
        '''vibration monitoring with user-friendly desktop and mobile apps that put the power of spectrum analysis in your handstestimonials of the company'''
        '''Our Mission: Elevate Asset Performance for the Power Operators with Modern Tools That Plant Employees Loveontact information of company and location'''
        '''privacy policy of company, log data, cookies, double click cookie, behavioural remarketing, service providers, security, international transfer'''
],
)


routes = [AssetSense]

import os
from semantic_router.encoders import CohereEncoder, OpenAIEncoder

# for Cohere
os.environ["COHERE_API_KEY"] = "4Lh5iCY5nc63ep5c4E9KLIFp1maSab4dAwgtAe5h"
encoder = CohereEncoder()

# # or for OpenAI
# os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
# encoder = OpenAIEncoder()
r1 = RouteLayer(encoder=encoder, routes=routes)
print("routing completed")
DB_FAISS_PATH = 'demovs3'
template = """Answer the question based on the context(delimited by <ctx> </ctx>) below and use the instructions mentioned(delimited by <inst> </inst>).

<inst>
Use the following pieces of information to answer the user's question.
Only use the context to answer the given question. Never generate answer on your own.
Never assume any information on your own. It is okay to return "No Answer", if the context is empty
The answer should be the summary context by using not more than 300 words. 
</inst>

<ctx>
{context}
</ctx>

Question: {question}

Answer:
"""


# Analyse the context carefully and understand the details that are provided in the context.
# Analyse and understand the question carefully.
# Based on the question asked search for the exact answer for the question in the context.
# If there is no context that could be the answer for the question asked just say that You have no information regarding that.
# Don't mention that company will return your money back. 
def set_custom_prompt():
    prompt = PromptTemplate(template=template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    retriever=db.as_retriever( search_type="mmr")
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_name_or_path = "TheBloke/Llama-2-7B-chat-GGUF"
model_basename = "llama-2-7b-chat.Q4_K_M.gguf"
MODEL_PATH ="/Users/developer/Documents/srija/llama-2-7b-chat.Q4_K_M.gguf"

def load_llm():

    n_gpu_layers = 33
    n_batch = 512
    llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx = 4096,
    max_new_tokens=500
)
    cl.user_session.set("llm", llm)
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'mps'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    retriever=db.as_retriever(search_kwargs={"k": 5} )
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa,retriever
faithfulness_chain = RagasEvaluatorChain(metric=faithfulness) 
answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy) 
context_rel_chain = RagasEvaluatorChain(metric=context_precision) 
# context_recall_chain = RagasEvaluatorChain(metric=context_recall)
# answer_correctness=RagasEvaluatorChain(metric=answer_correctness)
#answer_semantic_similarity=RagasEvaluatorChain(metric=answer_semantic_similarity)
# chain2,retriever2 = qa_bot()
@cl.on_chat_start
async def start():
    chain,retriever = qa_bot()
#     questions = [
#     "What is AssetSense?",
#     "Give Testimonials Of AssetSense Company?"
#     "What specific technologies does AssetSense utilize to ensure immediate value realization and rapid ROI?",
#     "According to the content, how does AssetSense support power plant executives in making informed decisions?",
#     "Can you explain how AssetSense's platform automation helps in capturing vibration data and performing predictive maintenance?",
#     "How does AssetSense cater to different types of power generation, such as fossil fuel, hydro, wind, and biofuel, as mentioned in the content?",
#     "What challenges do power plants face in terms of integrating digital technologies with existing systems, and how does AssetSense overcome these challenges?",
#     "How does AssetSense address the limitations of traditional asset monitoring systems, as highlighted in the content?",
#     "What is the role of mobile apps in AssetSense, and how do they contribute to the ease of completing key tasks for users?",
#     "Could you elaborate on the benefits of AssetSense's plug-and-play integrations with key systems like ERPs, historians, and enterprise asset management tools?",
#     "In what ways does AssetSense enhance plant efficiencies, and how does it contribute to the overall reliability of equipment in power generation?",
#     "What is the significance of AssetSense's private cloud in terms of security, performance, and reliability, as mentioned in the content?",
#     "How does AssetSense assist power plant managers, maintenance crews, and operators in monitoring equipment health and improving plant performance?",
#     "What is the significance of digital transformation for power generation companies in the context of market conditions and regulatory demands, as per the World Economic Forum and Deloitte survey?",
#     "According to the content, why is Asset Performance Management considered the single most valuable initiative for the electricity industry?",
#     "Can you provide examples of real-time insights that power plant executives can gain from using AssetSense, and how do these insights contribute to improving asset reliability?",
#     "What role does AssetSense play in elevating fleet-wide performance, and how does it contribute to lowering IT costs for power generation companies?"
# ]
#     print("questons")
#     print("questions")
#     # ground_truths = [["The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."],
#     #                 ["The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."],
#     # #                 ["The president asked Congress to pass proven measures to reduce gun violence."]]
#     answers = []
#     contexts = []
    
#     # Inference
#     for query in questions:
#       answers.append(chain.invoke(query)['result'])
#       ret_docs = retriever.get_relevant_documents(query)
#       contexts.append([docs.page_content for docs in ret_docs])
#     print("answers contexts created")
    
#     # To dict
#     data = {
#         "question": questions,
#         "answer": answers,
#         "contexts": contexts,
#     # "ground_truths": ground_truths
#     }
#     print("Data created")
    
#     # Convert dict to dataset
#     dataset = Dataset.from_dict(data)
#     print("dataset created")
#     print("dataset created")
#     print("dataset created")
    
#     result = evaluate(
#         dataset = dataset,
#         metrics=[
#             context_precision,
#             faithfulness,
#             answer_relevancy,
#             harmfulness
#         ],
#     )
    
#     print(result)
#     df = result.to_pandas()
#     print(df)
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
        answer = res["result"]
        sources = res["source_documents"]
        print(sources)
        eval_result = faithfulness_chain.invoke(res) 
        fres=""
        fres= fres+"Faithfulness: "+str(eval_result["faithfulness_score"])+"  "
    
    
        eval_result = answer_rel_chain.invoke(res) 
        fres = fres+"Answer Relevancy:"+str(eval_result["answer_relevancy_score"])+"  "
    
        eval_result = context_rel_chain.invoke(res) 
        fres= fres+"Context Precision: "+str(eval_result["context_precision_score"])+"  "

    else:
        answer = "Information not provided!"

    await cl.Message(content=answer+"\n\nSCORE\n "+fres).send()