# 📌 元知识蒸馏编码器训练

## 项目简介
```text
该代码文件夹包含元知识蒸馏编码器训练（以T5模型为例）所有的代码，用户可以根据下面快速运行的参考环境和依赖快速启动编码器训练流程。
```

## 快速运行
```text
1. 环境：Python 3.13.9; Torch 2.6.0+cu124; Transformers 5.5.4; Pandas 2.3.3
2. 安装依赖(bash)：
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install transformer
   pip install pandas
3. python main.py 启动训练流程
```

## 项目结构
```text
├── meta_kd/              
│   ├── base_t5_models.py           # 模型架构定义（包含层级划分逻辑，教师/学生/辅助/元学习器模型）
│   ├── hierarchical_loss.py        # 师生层级损失函数
│   ├── main.py                     # 模型训练主入口
│   ├── meta_distill_strategy.py    # 元蒸馏策略
│   ├── README.md                   
│   ├── use_student.py              # 模型推理示例脚本
│   └── weighted_aggregation.py     # 加权聚合算法
```

## 训练结果
```text
迭代500次，结果如下：
✅ 最优综合三层损失: 0.3307
✅ 最优模型出现在迭代: 455
✅ 模型保存为: t5_student.pth
```

## 模型更改与数据集替换
```text
1. 所有文件import时，将transformer库中的T5相关模型更换为目标模型
2. base_t5_models.py中需自定义教师/学生/辅助模型初始化架构，注意目标教师模型的层数并更改编码器大层划分的逻辑
3. main.py中更改tokenizer、teacher的模型载入路径，更改texts、val_csv_path以加载你自己的数据
```
