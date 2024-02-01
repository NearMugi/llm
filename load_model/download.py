import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

startTime = time.time()

model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"

print("+" * 10)
print("load tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("TIME : %s s" % round(time.time() - startTime, 1))

print("+" * 10)
print("load model")
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        torch_dtype=torch.float16
    )
print("TIME : %s s" % round(time.time() - startTime, 1))

print("+" * 10)
print("save tokenizer")
tokenizer.save_pretrained("./download/tokenizer")
print("TIME : %s s" % round(time.time() - startTime, 1))

print("+" * 10)
print("save model")
model.save_pretrained(
    "./download//model",
    overwrite_output_dir=True
    )

print("TOTAL TIME : %s s" % round(time.time() - startTime, 1))
