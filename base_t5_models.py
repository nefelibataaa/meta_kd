import torch
import torch.nn as nn
from transformers import T5Model, T5Config

# 编码器大层划分
def get_encoder_block_output(hidden_states, model_type):
    layer_feats = hidden_states[1:]
    block_feats = []
    if model_type == "teacher":
        block_feats = [layer_feats[7], layer_feats[15], layer_feats[23]]
    elif model_type == "aux":
        block_feats = [layer_feats[3], layer_feats[7], layer_feats[11]]
    elif model_type == "student":
        block_feats = [layer_feats[1], layer_feats[3], layer_feats[5]]
    return block_feats

# 教师T5（24层编码器）
class T5Teacher(nn.Module):
    def __init__(self, model_name="/home/xyq/T5"):
        super().__init__()
        self.t5 = T5Model.from_pretrained(model_name)
        self.config = self.t5.config
        self.config.num_layers = 24

    def forward(self, input_ids, attention_mask):
        outputs = self.t5.encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        return get_encoder_block_output(outputs.hidden_states, "teacher")

# 学生T5（6层编码器）
class T5Student(nn.Module):
    def __init__(self, teacher_config):
        super().__init__()
        student_config = T5Config(
            vocab_size=teacher_config.vocab_size,
            d_model=teacher_config.d_model,
            d_kv=teacher_config.d_kv,
            num_layers=6,
            num_heads=teacher_config.num_heads,
            d_ff=teacher_config.d_ff,
        )
        self.t5 = T5Model(student_config)
        self.config = student_config

    def forward(self, input_ids, attention_mask):
        outputs = self.t5.encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        return get_encoder_block_output(outputs.hidden_states, "student")

# 辅助T5（12层编码器）
class T5AuxModel(nn.Module):
    def __init__(self, teacher_config):
        super().__init__()
        aux_config = T5Config(
            vocab_size=teacher_config.vocab_size,
            d_model=teacher_config.d_model,
            d_kv=teacher_config.d_kv,
            num_layers=12,
            num_heads=teacher_config.num_heads,
            d_ff=teacher_config.d_ff,
        )
        self.t5 = T5Model(aux_config)
        self.config = aux_config

    def forward(self, teacher_block_features):
        aux_block_feats = []
        for feat in teacher_block_features:
            out = self.t5.encoder(inputs_embeds=feat, output_hidden_states=True)
            aux_block_feats.append(out.hidden_states[-1])
        return aux_block_feats

# 元学习器
class MetaLearnerMLP(nn.Module):
    """轻量MLP元学习器，学生/教师各实例化一个"""
    def __init__(self, in_dim=3, out_dim=3, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.mlp(x)
        return 1 + self.sigmoid(out) * 1  # 归一化 [1,2]
