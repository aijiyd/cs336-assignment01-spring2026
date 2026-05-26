from typing import Iterable
import pickle
import regex as re
import numpy as np


# 正则表达式，以符号为分隔切割文本
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
compiled_pat = re.compile(GPT2_PAT)
# 生成一个元组，里面有 256个基础字节
BYTES_TOKENS = tuple(bytes([i]) for i in range(256))

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)} # 处理合并优先级
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        self.special_tokens = special_tokens or []
        self.special_token_bytes = [token.encode("utf-8") for token in self.special_tokens]
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.token_to_id:
                new_id = len(vocab)
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id

        self.special_pattern = self._build_special_pattern()
        self.special_token_to_id = {token_bytes: self.token_to_id[token_bytes] for token_bytes in self.special_token_bytes}

    @classmethod
    def from_file(cls, vocab_path: str, merge_path: str, special_tokens: list[str]):
        with open(vocab_path, "rb") as f:
            vocab_content = pickle.load(f)
        
        # 做格式修改
        vocab = {}
        for key, value in vocab_content.items():
            id = int(key)
            if isinstance(value, str):
                value = value.encode("utf-8")
            vocab[id] = value

        with open(merge_path, "rb") as f:
            merge_content = pickle.load(f)

        merges = []
        for x, y in merge_content:
            if isinstance(x, str):
                x = x.encode("utf-8")
            if isinstance(y, str):
                y = y.encode("utf-8")
            merges.append((x, y))
        return cls(vocab, merges, special_tokens)

    def _build_special_pattern(self):
        if not self.special_tokens:
            return None
        escaped_tokens = [re.escape(tok) for tok in self.special_tokens]
        escaped_tokens = sorted(escaped_tokens, key=len, reverse=True)
        pattern = re.compile("(" + "|".join(escaped_tokens) + ")")  # ty:ignore[no-matching-overload]
        return pattern

    def _merge(self, tokens: list[bytes]):
        candidates = []
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merge_ranks:
                candidates.append((self.merge_ranks[pair], i, pair)) # 得到 候选 合并对：(rank, position, pair)
        
        if not candidates:
            return tokens, False
        
        _, i, pair = min(candidates)
        merge_token = pair[0] + pair[1]
        new_token = tokens[:i] + [merge_token] + tokens[i+2:]
        return new_token, True

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        
        if not self.special_tokens:
            parts = [text]
        else:
            parts = self.special_pattern.split(text)
            parts = [part for part in parts if part]
        
        all_ids = []

        for part in parts:
            # 编码特殊标记
            if part in self.special_tokens:
                token_bytes = part.encode("utf-8")
                all_ids.append(self.special_token_to_id[token_bytes])
            else:
                """只处理普通文本块,把普通文本块切成 pretokens"""
                pretokens = []
                for match in compiled_pat.finditer(part):
                    pretokens.append(match.group(0))
                
                text_ids = []
                for pretoken in pretokens:
                    seq_tokens = [BYTES_TOKENS[i] for i in pretoken.encode("utf-8")]
                    while True:
                        seq_tokens, changed = self._merge(seq_tokens)
                        if not changed:
                            break
                    token_ids = [self.token_to_id[token] for token in seq_tokens]
                    all_ids.extend(token_ids)

        return all_ids

    def encode_iterable(self, iterable: Iterable[str]):
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def decode(self, ids: list[int]):
        seq_bytes = b"".join(self.vocab[i] for i in ids)
        return seq_bytes.decode("utf-8", errors="replace")



if __name__ == "__main__":
    tokenizer = BPETokenizer.from_file(
        "data/tinystories_vocab.pkl",
        "data/tinystories_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
    
    for input_path, output_path in [
        ("data/TinyStoriesV2-GPT4-train.txt", "data/tinystories_train_tokens.npy"),
        ("data/TinyStoriesV2-GPT4-valid.txt", "data/tinystories_valid_tokens.npy"),
    ]:
        print(f"Loading {input_path}", flush=True)
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Encoding {input_path}, chars={len(text)}", flush=True)
        ids = tokenizer.encode(text)
        print(f"Saving {output_path}, tokens={len(ids)}", flush=True)
        arr = np.array(ids, dtype=np.uint16)
        np.save(output_path, arr)
    
        print(input_path, "->", output_path, len(arr))

        

