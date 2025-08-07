from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = "./"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).cuda()

inputs = tokenizer("Explain the basics of quantum entanglement.", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
