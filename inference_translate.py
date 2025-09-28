def translate(model, tokenizer, sample_text):
    batch = tokenizer([sample_text], return_tensors="pt")
    generated_ids = model.generate(**batch)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

if __name__ == '__main__':
    from transformers import AutoTokenizer, MarianMTModel

    src = "en"  
    trg = "hi"  

    model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sample_text = "Hi How are you?"
    batch = tokenizer([sample_text], return_tensors="pt")

    generated_ids = model.generate(**batch)
    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])