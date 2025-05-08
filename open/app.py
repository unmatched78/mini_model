from transformers import GPTNeoForCausalLM, GPT2Tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  
model = GPTNeoForCausalLM.from_pretrained(model_name)  
tokenizer = GPT2Tokenizer.from_pretrained(model_name)  

input_text = "Once upon a time"  
input_ids = tokenizer.encode(input_text, return_tensors='pt')  

output = model.generate(input_ids, max_length=50)  
print(tokenizer.decode(output[0], skip_special_tokens=True))