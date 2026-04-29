import torch
from transformers import T5Tokenizer, T5EncoderModel, T5Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("/home/xyq/T5")

config = T5Config.from_pretrained("/home/xyq/T5")
config.num_layers = 6
model = T5EncoderModel(config).to(DEVICE)

MODEL_PATH = "/home/xyq/workspace/meta_kd/t5_encoder_only.pth"
raw_state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
new_state_dict = {}

for k, v in raw_state_dict.items():
    new_key = "encoder." + k
    
    if k == "embed_tokens.weight":
        new_state_dict["shared.weight"] = v
        
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()
print("✅ 权重映射成功，模型已就绪！")

text = "这是一个测试句子"
inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)
    print("编码成功！输出维度：", outputs.last_hidden_state)
