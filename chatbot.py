#!/usr/bin/env python3
# chatbot.py

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

def main():
    # 1. Diccionario de ventas en memoria
    ventas = {
        "Zapatos": 120,
        "Camisetas": 75,
        "Pantalones": 50,
        "Sombreros": 30
    }

    # 2. Transformar cada ítem en un documento de texto
    docs = [f"Producto: {producto}, Ventas: {cantidad}" 
            for producto, cantidad in ventas.items()]

    # 3. Generar embeddings y montar el índice FAISS
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_index = FAISS.from_texts(docs, embeddings)

    # 4. Configurar el chain de QA
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=faiss_index.as_retriever(search_kwargs={"k": 2})
    )

    # 5. Bucle interactivo de consulta
    print("🤖 Chatbot de Ventas (escribe 'salir' para terminar)")
    while True:
        pregunta = input("\n¿Qué quieres saber de las ventas? ")
        if pregunta.strip().lower() in ("salir", "exit", "quit"):
            print("👋 ¡Hasta luego!")
            break
        respuesta = qa.run(pregunta)
        print(f"💡 {respuesta}")

if __name__ == "__main__":
    # Asegúrate de exportar tu clave antes de ejecutar:
    # export OPENAI_API_KEY="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"
    if not os.getenv("OPENAI_API_KEY"):
        print("❗️ Define la variable OPENAI_API_KEY antes de ejecutar.")
    else:
        main()
