import json

from transformers import AutoConfig, AutoModelForMultimodalLM
from accelerate import init_empty_weights

from dotenv import load_dotenv

load_dotenv()

config = AutoConfig.from_pretrained("google/diffusiongemma-26B-A4B-it")
with init_empty_weights():
    model = AutoModelForMultimodalLM.from_config(config)

def describe(module):
    node = {"type": type(module).__name__}
    extra = module.extra_repr()
    if extra:
        node["config"] = extra
    children = {name: describe(child) for name, child in module.named_children()}
    if children:
        node["children"] = children
    return node


decoder = model.model.decoder
print(decoder)

with open("decoder_layers.json", "w") as f:
    json.dump(describe(decoder), f, indent=2)
