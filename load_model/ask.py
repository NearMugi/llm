import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import time

startTime = time.time()
print("+" * 10)
print("load tokenizer")
tokenizer = AutoTokenizer.from_pretrained("./download/tokenizer")
print("TIME : %s s" % round(time.time() - startTime, 1))
print("+" * 10)
print("load model")
model = LlamaForCausalLM.from_pretrained("./download/model")
print("TIME : %s s" % round(time.time() - startTime, 1))

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
text = "エラトステネスの篩についてサンプルコードを示し、解説してください。"

print("+" * 10)
print("prompt")
prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)
print("TIME : %s s" % round(time.time() - startTime, 1))

with torch.no_grad():
    print("+" * 10)
    print("tokenizer.encode")
    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt")
    print("TIME : %s s" % round(time.time() - startTime, 1))

    print("+" * 10)
    print("model.generate")
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=64,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("TIME : %s s" % round(time.time() - startTime, 1))

    print("+" * 10)
    print("tokenizer.decode")
    output = tokenizer.decode(
        output_ids.tolist()[0][token_ids.size(1):],
        skip_special_tokens=True)

    print(output)

print("TOTAL TIME : %s s" % round(time.time() - startTime, 1))
