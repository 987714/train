import paddlenlp

tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

text = "今天天气真好！"
result = []
print(text)
encoded_inputs = tokenizer(text=text, max_length=512)
print(encoded_inputs)
print(encoded_inputs["input_ids"])
print(encoded_inputs["token_type_ids"])
result += [encoded_inputs["input_ids"], encoded_inputs["token_type_ids"]]
print('------------------------------------')

print(result)