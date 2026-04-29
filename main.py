import torch
import pandas as pd
from transformers import T5Tokenizer
from base_t5_models import T5Teacher, T5Student
from hierarchical_loss import T5HierarchicalLossCalculator
from meta_distill_strategy import T5BiMetaDistillStrategy
from weighted_aggregation import T5WeightedAggregationDistill

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("/home/xyq/T5")

# 模型初始化
teacher = T5Teacher("/home/xyq/T5").to(DEVICE)
student = T5Student(teacher.config).to(DEVICE)

# 模块初始化
loss_calc = T5HierarchicalLossCalculator()
strategy = T5BiMetaDistillStrategy(teacher.config)
distiller = T5WeightedAggregationDistill(student)

# 数据
MAX_LEN = 23
texts = ["74,0,0.000000,TCP,1.0,1.0"]
inputs = tokenizer(texts, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="pt").to(DEVICE)
# inputs = tokenizer(texts, padding=True, return_tensors="pt").to(DEVICE)

val_csv_path = "/home/xyq/workspace/data/meta_data/meta_data.csv"
df = pd.read_csv(val_csv_path)
n_sample = min(100, len(df))
df_sample = df.sample(n=n_sample, random_state=42)

val_texts = []
for _, row in df_sample.iterrows():
    line = f"{row['packet_size']},{row['direction']},{row['inter_arrival_time']},{row['protocol_type']},{row['up_pkt_ratio']},{row['up_byte_ratio']}"
    val_texts.append(line)

val_inputs = tokenizer(val_texts, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="pt").to(DEVICE)
# val_inputs = tokenizer(val_texts, padding=True, return_tensors="pt").to(DEVICE)

if __name__ == "__main__":
    print("="*60)
    print("📌 T5编码器元知识蒸馏")
    print("学生元学习器：Lms→W_S｜教师元学习器：Lmt→W_T")
    print("="*60)
    
    student_final, loss_hist = distiller.iterative_train(
        loss_calc, strategy, teacher, inputs.input_ids, inputs.attention_mask, val_inputs
    )
    
    print("🎉 学生模型训练完成！")
