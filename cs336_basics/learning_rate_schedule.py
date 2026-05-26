import math
def get_lr_cosine_schedule(
    t: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if t < warmup_iters:
        alpha = t / warmup_iters * max_learning_rate
    elif t <= cosine_cycle_iters and t >= warmup_iters:
        alpha = min_learning_rate + 0.5 * (1 + math.cos(math.pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        alpha = min_learning_rate
    return alpha
