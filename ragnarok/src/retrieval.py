from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


def preparar_bm25(documentos):
    tokenized = [word_tokenize(doc.lower()) for doc in documentos]
    return BM25Okapi(tokenized), documentos


def recuperar_documentos(pergunta, bm25, documentos, k=3):
    tokenized_query = word_tokenize(pergunta.lower())
    scores = bm25.get_scores(tokenized_query)
    topk = sorted(range(len(scores)),
                  key=lambda i: scores[i], reverse=True)[:k]
    return [documentos[i] for i in topk]
