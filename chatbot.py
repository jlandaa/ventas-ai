#!/usr/bin/env python3
from dotenv import load_dotenv
import os, re
from openai.error                   import RateLimitError
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms       import OpenAI
from langchain.chains               import RetrievalQA

def respuesta_local(q, ventas):
    q = q.lower()
    if "menos ventas" in q:
        p,v = min(ventas.items(), key=lambda x:x[1])
        return f"El producto con menos ventas fue {p}, con {v} unidades vendidas."
    if "más ventas" in q or "mayor venta" in q:
        p,v = max(ventas.items(), key=lambda x:x[1])
        return f"El producto con más ventas fue {p}, con {v} unidades vendidas."
    return None

def main():
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        print("❗️ Define OPENAI_API_KEY en .env")
        return

    ventas = {"Zapatos":120,"Camisetas":75,"Pantalones":50,"Sombreros":30}
    docs   = [f"Producto: {p}, Ventas: {v}" for p,v in ventas.items()]

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    index      = FAISS.from_texts(docs, embeddings)

    llm = OpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=API_KEY,
        max_retries=0
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever(search_kwargs={"k":1})
    )

    print("🤖 Chatbot de Ventas (escribe 'salir' para terminar)")
    while True:
        pregunta = input("> ¿Qué quieres saber de las ventas? ").strip()
        if pregunta.lower() in ("salir","exit","quit"):
            print("👋 ¡Hasta luego!")
            break

        # 1) Local
        resp = respuesta_local(pregunta, ventas)
        if resp:
            print("💡", resp)
            continue

        # 2) LLM con 0 retries
        try:
            res = qa.invoke({"query": pregunta})
            print("💡", res["result"])
        except RateLimitError:
            print("⚠️ No se puede consultar a OpenAI, se acabó la cuota.")

if __name__ == "__main__":
    main()

