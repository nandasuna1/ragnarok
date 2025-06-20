import gradio as gr
import pandas as pd
import nltk
from ragnarok.src.retrieval import preparar_bm25, recuperar_documentos
from ragnarok.src.generation import gerar_resposta


# Baixar tokenizador
nltk.download("punkt")

# Carrega os dados
df = pd.read_csv("data/processed/clean_data.csv")
corpus = df["document"].tolist()

# Prepara o BM25
bm25, tokenized_corpus = preparar_bm25(corpus)

# Função usada pelo Gradio


def responder(pergunta, k=3):
    docs = recuperar_documentos(pergunta, bm25, corpus, k)
    return gerar_resposta(pergunta, docs)


# Cria a interface
demo = gr.Interface(
    fn=responder,
    inputs=[
        gr.Textbox(label="Type your question"),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Aviation incident accident report (RAG - NTSB)"
)

if __name__ == "__main__":
    # demo.launch(share=True)
    demo.launch()
