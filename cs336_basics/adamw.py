import math
from typing import Optional, Callable
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float=1e-3, weight_decay:float=0.01, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-8): 
        if lr < 0: 
            raise ValueError(f"Invalid learning rate: {lr}") 
        defaults = {
            "lr": lr, # 学习率
            "betas": betas, # 矩估计
            "weight_decay": weight_decay, # 权重衰减参数
            "eps": eps
        } 
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure() 
        for group in self.param_groups: 
            alpha = group["lr"] # Get the learning rate. 
            beta_1, beta_2 = group["betas"]
            lamda = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]: 
                if p.grad is None: 
                    continue  
                
                state = self.state[p] # Get state associated with p. 
                grad = p.grad.data # Get the gradient of loss with respect to p. 

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    
                # 旧状态
                t = state["t"]
                m = state["m"]
                v = state["v"]

                t = t + 1

                p.data -= alpha * lamda * p.data

                # 一阶矩估计
                m = beta_1 * m + (1 - beta_1) * grad
                # 二阶矩估计
                v = beta_2 * v + (1 - beta_2) * (grad ** 2)

                alpha_t = alpha * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                
                # 更新参数状态
                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss