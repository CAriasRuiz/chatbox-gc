import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

with open("normativa.txt", "r", encoding="utf-8") as f:
    documento = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(documento)

vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}  # Retorna los 4 fragmentos mas relevantes
)

prompt = ChatPromptTemplate.from_template("""Eres el Asistente Experto en Documentación Técnica y Normativa de la Gerencia de Compras de
Empresa Nacional del Petrroleo. Tu función es proporcionar respuestas precisas, formales y estrictamente basadas en la documentación oficial proporcionada..

Reglas estrictas:
1. Responderás única y exclusivamente utilizando la información contenida en el documento o contexto proporcionado. 
Si la respuesta no se encuentra en el texto, deberás indicar cortésmente: "Lo siento, pero la información solicitada no se encuentra 
en la documentación disponible. Te recomiendo contactarnos
  al mail a contacto_ger_compras@enap.cl".
2. Bajo ninguna circunstancia inventarás datos, fechas, nombres de leyes o procedimientos que no estén explícitamente escritos 
en el contexto.
3. Tu lenguaje debe ser profesional, objetivo y administrativo. Evita el uso de jerga informal o expresiones subjetivas.
4. Siempre que sea posible, menciona la sección, artículo o apartado del documento de donde extrajiste la información 
(ej. "Según el apartado 4.2 del manual...").

Contexto:
{context}

Pregunta del cliente: {question}

Respuesta:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def respond(message, history):
    response = rag_chain.invoke(message)
    return response

demo = gr.ChatInterface(
    fn=respond,
    title="Gerencia de Compras - Asistente Virtual",
    description="Bienvenido al asistene Virtual de la Gerencia de Compras de Enap",
    examples=[
        "¿Cuales son los niveles de aprobacion?",
       
    ],
)

if __name__ == "__main__":
    print(f"Documento cargado: {len(chunks)} fragmentos indexados")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme="soft"
    )