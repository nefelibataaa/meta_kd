import torch
import torch.nn as nn

class T5HierarchicalLossCalculator:
    def __init__(self):
        self.mse = nn.MSELoss(reduction='sum')
    
    def compute_initial_loss(self, teacher_feats, student_feats):
        block_losses = []

        for i in range(3):
            ft = teacher_feats[i]
            fs = student_feats[i]
            B, L, D = ft.shape
            loss = self.mse(ft, fs) / (B * L * D)
            block_losses.append(loss)
        
        return torch.stack(block_losses)
