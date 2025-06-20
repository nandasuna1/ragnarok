from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


def gerar_resposta(pergunta, documentos, max_tokens=256):
    contexto = " ".join(documentos)
    prompt = (
        f"Bellow you have context of one or more airplane accidents documents. "
        f"With this information answer the question clearly\n\n"
        f"ducuments:\n{contexto}\n\n"
        f"question: {pergunta}\n"
        f"answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
