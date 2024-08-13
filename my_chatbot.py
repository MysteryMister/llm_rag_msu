from fastapi import FastAPI
import uvicorn

from qdrant_client import QdrantClient

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


app = FastAPI()

qdrant_client = QdrantClient("localhost", port=6333, timeout=100)

hugging_face_pipeline = None
doc_store = None
tokenizer = None


@app.post('/search')
def search(query, sentences_number=5, score_threshold=0.8):
    retriever = doc_store.as_retriever(
        search_type='mmr', 
        search_kwargs={"k": int(sentences_number), "score_threshold": float(score_threshold)}
    )

    return retriever.get_relevant_documents('query: ' + query)


@app.post('/ask')
def search_and_generate(query, sentences_number=5, score_threshold=0.8, use_rag=True):
    def postprocess(generation):
        return generation[generation.find('GPT4 Correct Assistant:') + len('GPT4 Correct Assistant:'):].strip()

    relevant_docs = search(query, sentences_number, score_threshold)

    context = []
    for i, doc in enumerate(relevant_docs):
        context.append(f'{i + 1}. {doc.metadata["context"]}')
    prompt_context = '\n'.join(context)

    system_part = f"""You are a literary theorist who has the high expertise in the writings of George Orwell.
Your task is to answer the question about his dystopian novel '1984'.
Write everything you know about the question.
"""
    rag_part = f"""Use the following relevant information during the answer.

Relevant information:
{prompt_context}
""" 
    question_part = f"""Question: {query}"""

    # no RAG
    if not use_rag:
        prompt = '\n'.join([system_part, question_part])
    # use RAG
    else:
        prompt = '\n'.join([system_part, rag_part, question_part])

    chat = [{"role": "user", "content": prompt}]

    chain = hugging_face_pipeline | postprocess
    return chain.invoke(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))


if __name__ == '__main__':
    embeddings = HuggingFaceEmbeddings(
        model_name='intfloat/multilingual-e5-large',
        model_kwargs={"device": "cuda"}
    )

    qdrant_client = QdrantClient("localhost", port=6333, timeout=100)
    doc_store = Qdrant(
        qdrant_client, 
        embeddings=embeddings,
        collection_name='qdrant_novels_1984',
        content_payload_key='page_content',
        metadata_payload_key='metadata',
    )

    checkpoint = 'openchat/openchat-3.5-0106'

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    pipeline_kwargs = {
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 0.2,
        "top_k": 30,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "pad_token_id": 32000,
    }
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **pipeline_kwargs)
    hugging_face_pipeline = HuggingFacePipeline(pipeline=pipe)

    uvicorn.run(app)
