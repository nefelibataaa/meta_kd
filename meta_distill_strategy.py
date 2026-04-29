import torch
import torch.nn as nn
import torch.nn.functional as F
from base_t5_models import MetaLearnerMLP, T5AuxModel, T5Student

class T5BiMetaDistillStrategy:
    def __init__(self, teacher_config, beta=1e-3):
        self.beta = beta
        self.num_blocks = 3

        # 学生元学习器：输入3维层级损失 → 输出3维(Ws)
        self.meta_student = MetaLearnerMLP(in_dim=3, out_dim=3)
        # 教师元学习器：输入3维任务损失 → 输出3维(Wt)
        self.meta_teacher = MetaLearnerMLP(in_dim=3, out_dim=3)

        # 辅助模型
        self.aux_model = T5AuxModel(teacher_config)
        self.opt_aux = torch.optim.Adam(self.aux_model.parameters(), lr=1e-3)

    # 匹配损失 L_match
    def _match_loss(self, weight, target_loss):
        return -torch.mean((weight - target_loss) ** 2)

    # 正则损失 L_reg
    def _reg_loss(self, *weights):
        reg = 0
        for w in weights:
            reg += torch.norm(w, p=2)
        return reg

    # 学生元学习器：Lms损失 → 输出WS+WA
    def optimize_meta_student(self, L_mse):
        """
        论文Lms损失：
        Lms = Lmatch(WS,Lmse) + β*Lreg(WA,WS)
        输出：WS(学习权重)
        """
        opt = torch.optim.Adam(self.meta_student.parameters(), lr=1e-3)
        # 前向输出
        W_S = self.meta_student(L_mse.unsqueeze(0)).squeeze(0)

        # 计算Lms损失
        loss_match_S = self._match_loss(W_S, L_mse)
        loss_reg = self._reg_loss(W_S)
        L_ms = loss_match_S + self.beta * loss_reg

        # 优化更新
        opt.zero_grad()
        L_ms.backward()
        opt.step()
        return W_S.detach()

    # 辅助模型更新
    def update_aux_model(self, teacher_feats, W_S):
        aux_feats = self.aux_model(teacher_feats)
        losses = []
        for i in range(3):
            ft, fa = teacher_feats[i], aux_feats[i]
            B, L, D = ft.shape
            losses.append(F.mse_loss(fa, ft) / (B*L*D))
        total_loss = torch.sum(W_S * torch.stack(losses))
        self.opt_aux.zero_grad()
        total_loss.backward()
        self.opt_aux.step()
        return self.aux_model

    # 克隆学生前向
    def clone_student_forward(self, clone_student, input_ids, attention_mask):
        return clone_student(input_ids, attention_mask)

    # 教师元学习器：Lmt损失 → 输出WT
    def optimize_meta_teacher(self, L_task):
        """
        论文Lmt损失：
        Lmt = -1/LΣ(Wtl-Ltaskl)² + β*Lreg(WT)
        输出：WT(教学权重)
        """
        opt = torch.optim.Adam(self.meta_teacher.parameters(), lr=1e-3)
        # 前向输出
        W_T = self.meta_teacher(L_task.unsqueeze(0)).squeeze(0)

        # 计算Lmt损失
        loss_match_T = self._match_loss(W_T, L_task)
        loss_reg = self._reg_loss(W_T)
        L_mt = loss_match_T + self.beta * loss_reg

        # 优化更新
        opt.zero_grad()
        L_mt.backward()
        opt.step()
        return W_T.detach()

    # 教师特征调制
    def modulate_teacher(self, teacher_feats, W_T):
        return [teacher_feats[i] * W_T[i] for i in range(3)]
