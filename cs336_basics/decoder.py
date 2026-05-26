import torch
from cs336_basics.softmax import softmax

def decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    device: str,
    temperature: float=1.0,
    top_p: float | None=None,
    end_token: str="<|endoftext|>"
):
    prompt_token_ids = tokenizer.encode(prompt)
    end_token_ids = tokenizer.encode(end_token)
    assert len(end_token_ids) == 1
    end_token_id = end_token_ids[0]

    assert temperature >= 0
    assert len(prompt_token_ids) > 0

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = prompt_token_ids[-context_length:]
            inputs = torch.tensor(input_ids, dtype=torch.long, device=torch.device(device)).reshape(1, -1)
            logits = model(inputs) # (1, seq_len, vocab_size)
            # 取最后一个 token 用于预测下一个 token
            next_logits = logits[:, -1, :].reshape(-1)
            if temperature == 0:
                next_token_id = torch.argmax(next_logits).item()
            else:
                probs = softmax(next_logits / temperature, dim=-1)

                # 截断低概率
                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    remove_mask = cumulative_probs > top_p
                    remove_mask[1:] = remove_mask[:-1].clone() # 右移一位，保留第一个让累计概率超过 top_p 的 token，让该位置为 false
                    remove_mask[0] = False # 第一个token必须保留
                    sorted_probs[remove_mask] = 0.0
                    sorted_probs /= sorted_probs.sum()

                    # 将排序后的概率采样成原 token id
                    next_token_sorted_pos = torch.multinomial(sorted_probs, num_samples=1)
                    next_token_id = sorted_indices[next_token_sorted_pos].item()
                else:
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                
            if next_token_id == end_token_id:
                break
            prompt_token_ids.append(next_token_id)

    output_text = tokenizer.decode(prompt_token_ids)
    return output_text
