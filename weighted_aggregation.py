import torch
import torch.nn.functional as F
from base_t5_models import T5Student

class T5WeightedAggregationDistill:
    def __init__(self, student_model, lr=1e-3):
        self.student = student_model
        self.opt = torch.optim.Adam(student_model.parameters(), lr=lr)

    def aggregate_knowledge(self, aux_feats, W_S):
        agg_feat = sum([W_S[i] * aux_feats[i] for i in range(3)])
        return agg_feat

    def update_real_student(self, agg_feat, student_feats):
        student_final = student_feats[-1]
        B, L, D = student_final.shape
        loss = F.mse_loss(student_final, agg_feat) / (B*L*D)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def iterative_train(self, loss_calc, strategy, teacher, input_ids, attention_mask, val_inputs, max_iter=500):
        loss_history = []
        prev_loss = None
        self.best_total_loss = float('inf')
        self.best_iteration = 0

        # 归一化到 [-1, 1]
        def normalize(feat):
            mean = feat.mean(dim=-1, keepdim=True)
            std = feat.std(dim=-1, keepdim=True)
            feat = (feat - mean) / (std + 1e-8)
            return torch.tanh(feat)

        for iter_idx in range(max_iter):
            teacher_feats = teacher(input_ids, attention_mask)
            teacher_feats = [feat.detach() for feat in teacher_feats]
            teacher_feats = [normalize(f) for f in teacher_feats]

            if iter_idx == 0:
                student_feats = self.student(input_ids, attention_mask)
                student_feats = [normalize(f) for f in student_feats]
                L_mse = loss_calc.compute_initial_loss(teacher_feats, student_feats)
                current_meta_loss = L_mse
            else:
                current_meta_loss = prev_loss

            # 元学习优化
            W_S = strategy.optimize_meta_student(current_meta_loss)
            aux_model = strategy.update_aux_model(teacher_feats, W_S)

            # 克隆学生验证
            clone_student = T5Student(self.student.config)
            clone_student.load_state_dict(self.student.state_dict())
            clone_feats = strategy.clone_student_forward(clone_student, val_inputs.input_ids, val_inputs.attention_mask)
            clone_feats = [normalize(feat.detach()) for feat in clone_feats]
            clone_feats_pooled = [f.mean(dim=0, keepdim=True) for f in clone_feats]

            # 教师元学习
            feats = aux_model(teacher_feats)
            feats = [normalize(f) for f in feats]
            L_task = loss_calc.compute_initial_loss(feats, clone_feats_pooled)
            W_T = strategy.optimize_meta_teacher(L_task)
            modulated_teacher = strategy.modulate_teacher(teacher_feats, W_T)
            modulated_teacher = [normalize(f) for f in modulated_teacher]

            # 更新学生模型
            student_feats = self.student(input_ids, attention_mask)
            student_feats = [normalize(f) for f in student_feats]
            aux_feats = aux_model(modulated_teacher)
            aux_feats = [normalize(f) for f in aux_feats]
            agg_feat = self.aggregate_knowledge(aux_feats, W_S)
            self.update_real_student(agg_feat, student_feats)

            # 验证集推理
            val_student_feats = self.student(val_inputs.input_ids, val_inputs.attention_mask)
            val_student_feats = [normalize(f) for f in val_student_feats]
            val_student_feats_pooled = [f.mean(dim=0, keepdim=True) for f in val_student_feats]

            # 最终损失
            final_loss = loss_calc.compute_initial_loss(modulated_teacher, val_student_feats_pooled)
            prev_loss = final_loss

            # 记录损失
            if iter_idx == 0:
                loss_history.append(L_mse)
            else:
                loss_history.append(final_loss)
            
            current_total = final_loss.mean().item()

            # 更小则保存最优模型
            if current_total < self.best_total_loss:
                self.best_total_loss = current_total
                self.best_iteration = iter_idx + 1
                torch.save(self.student.state_dict(), "best_t5_student.pth")
                print(f"\n🎉 新最优模型 | 迭代 {iter_idx+1} | 综合三层损失: {current_total:.4f}")

            print(f"\n📌 迭代 {iter_idx+1}/{max_iter}")
            print(f"├─ 三层损失: [{final_loss[0]:.4f}, {final_loss[1]:.4f}, {final_loss[2]:.4f}]")
            print(f"├─ 综合平均损失: {current_total:.4f}")
            print(f"└─ 历史最优损失: {self.best_total_loss:.4f} (第{self.best_iteration}轮)")

        # 训练结束总结
        print("\n" + "="*65)
        print("🎯 训练完成！")
        print(f"✅ 最优综合三层损失: {self.best_total_loss:.4f}")
        print(f"✅ 最优模型出现在迭代: {self.best_iteration}")
        print(f"✅ 模型已保存为: best_t5_student.pth")
        print("="*65)

        return self.student, loss_history
