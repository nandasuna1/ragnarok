# ✈️ Ragnarok: Sistema RAG para Incidentes Aéreos da NTSB

Este projeto implementa uma solução de **Retrieval-Augmented Generation (RAG)** para responder perguntas com base nos relatórios da **NTSB (National Transportation Safety Board)** sobre incidentes e acidentes aéreos nos Estados Unidos.

> Desenvolvido como atividade da disciplina de Mineração de Texto

## 🔍 Visão Geral

**Retrieval-Augmented Generation (RAG)** é uma abordagem que combina recuperação de informações (IR) com geração de linguagem natural (NLG). Nosso sistema utiliza:

- **BM25** (via `rank_bm25`) para buscar documentos relevantes.
- **flan-t5-base** da Hugging Face para gerar respostas baseadas nos documentos.

## 🤖 Como Usar

1. Clone o repositório do git

2. Crie um ambiente virtual (opcional)

```
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Instale as dependencias

```
pip install -r requirements.txt
```

4. execute a aplicação

```
python app.py
```

A interface Gradio será aberta em http://127.0.0.1:7860.

Exemplo de pergunta

## 🧱 Arquitetura da Solução

A solução foi estruturada em **três etapas principais**, seguindo o fluxo de um sistema RAG clássico:

### 1. **Preprocessamento e Tratamento dos Dados**

- Os dados foram carregados de um arquivo `.csv` com os relatórios da NTSB.
- Realizou-se a limpeza textual e a unificação de campos relevantes (como `date`, `location`, `probable_cause`, etc.) em um campo `document`.
- Os documentos foram transformados para letras minúsculas e tokenizados com `nltk.word_tokenize`.

### 2. **Recuperação de Documentos com BM25**

- Foi utilizado o modelo **BM25Okapi** da biblioteca `rank_bm25` para indexar os documentos.
- Uma função `recuperar_documentos(pergunta, bm25, corpus)` retorna os documentos mais relevantes (top-3) com base na similaridade da pergunta com os tokens dos documentos.

### 3. **Geração de Resposta com LLM**

- A geração da resposta é feita com o modelo `flan-t5-base` da Hugging Face (`transformers`).
- A função `gerar_resposta(pergunta, documentos)` monta um prompt contendo os documentos relevantes e a pergunta, e o envia ao modelo para gerar uma resposta textual.
- A geração é controlada por `max_tokens` para respeitar o limite de entrada do modelo.

### 4. **Interface com Gradio**

- Uma interface simples com `Gradio` permite testar o sistema com perguntas personalizadas.
- A interface exibe:

  - Pergunta do usuário
  - Resposta gerada pelo modelo

## ⚠️ Dificuldades Enfrentadas

- **Dependências do NLTK:** O módulo `punkt` apresentou problemas de localização ao usar `venv`, sendo necessário reinstalar manualmente os recursos e ajustar os caminhos.
- **Limite de tokens do modelo:** O `flan-t5-base` tem um limite de entrada, então em perguntas mais amplas ou com documentos longos, parte do contexto pode ser truncada.
- **Formatos dos documentos:** Alguns documentos da NTSB são mal formatados ou contêm valores nulos em campos importantes, o que exigiu cuidados na concatenação.

## 📌 Limitações da Solução

- O modelo não tem conhecimento factual externo, ele apenas "imita" uma resposta com base nos documentos fornecidos.
- O sistema não realiza **verificação de consistência** entre a resposta gerada e os documentos (ex: pode gerar uma resposta plausível mas não textual do documento).
- Não há suporte a múltiplos idiomas nem análise semântica profunda (ex: inferência de causas indiretas).
- A solução depende da qualidade da recuperação: se o BM25 não recuperar bons documentos, a geração será menos precisa.
- Ainda não foi feita a avaliação quantitativa (ex: ROUGE, BLEU ou similaridade) da qualidade das respostas — apenas qualitativa.

## 💡 Exemplos de Perguntas de Teste

1. What caused the accident during takeoff?
2. Were there any fatalities in the accident near Texas?
3. What was the weather like during the crash in New Mexico?
4. Did pilot error contribute to the accident in Florida?
5. what cause the accident in vero beach accident?
6. What was the probable cause of the accident on May 30, 2023?
7. How severe were the injuries reported in the New Mexico accident?
8. Were there any accidents during night flights?
9. Did any accidents involve mechanical failure?
10. What was the probable cause of the accident with most causalities?
