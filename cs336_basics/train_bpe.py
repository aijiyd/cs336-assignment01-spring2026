from typing import BinaryIO
import multiprocessing as mp
import os 
import pickle 
from collections import Counter, defaultdict
from tqdm import tqdm
import regex as re


# 正则表达式，以符号为分隔切割文本
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 生成一个元组，里面有 256个基础字节
BYTES_TOKENS = tuple(bytes([i]) for i in range(256))

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
    
def load_file(file_name: str):
    """从文件中加载文本"""
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read()
        return text

def word_to_byte_tuple(token: str):
    """将字符串先用 utf-8编码成数字串再转化为字节"""
    encoded = token.encode("utf-8")
    return tuple(BYTES_TOKENS[b] for b in encoded)

def pairs_in_word(word:tuple[bytes, ...]) -> Counter[tuple[bytes, bytes]]:
    """
    返回一个词内部所有相邻 pair 的频次
        (b'a', b'b', b'a', b'b') -> {(b'a', b'b'): 2, (b'b', b'a'): 1}
    """
    # 不存在相邻对
    if len(word) < 2:
        return Counter()
    return Counter(zip(word, word[1:]))

def build_pair_index(word_counts: Counter[tuple[bytes, bytes]]):
    """
    构建：
    1. pair_counts: 每个 pair 在整个语料中出现的总频次
    2. pair_to_words: 每个 pair 出现在哪些词里
       这里 value 用 Counter 而不是 set，是为了处理：
       不同旧词 merge 后变成同一个 new_word 的碰撞情况
    """
    pair_counts = Counter()
    pair_to_words = defaultdict(Counter)

    for word, freq in word_counts.items():
        word_pair_counts = pairs_in_word(word)
        for pair, multiplicity in word_pair_counts.items():
            pair_counts[pair] += multiplicity * freq
            pair_to_words[pair][word] += 1

    return pair_counts, pair_to_words

def merge_word(word: tuple[bytes, ...], best_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged_token = best_pair[0] + best_pair[1]
    new_word = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and (word[i], word[i + 1]) == best_pair:
            new_word.append(merged_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def apply_merge(
    word_counts: Counter[tuple[bytes, ...]],
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_words: dict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]],
    best_pair: tuple[bytes, bytes],
):
    # 要处理的词
    affected_words = list(pair_to_words.get(best_pair, {}).keys())
    if not affected_words:
        return word_counts, pair_counts, pair_to_words
    
    affected_freqs = {word: word_counts[word] for word in affected_words}

    for old_word in affected_words:
        freq = affected_freqs[old_word]

        old_word_counts = pairs_in_word(old_word)
        for pair, counts in old_word_counts.items():
            pair_counts[pair] -= counts * freq
            if pair_counts[pair] <= 0:
                del pair_counts[pair]

            pair_to_words[pair][old_word] -= 1
            if pair_to_words[pair][old_word] <= 0:
                del pair_to_words[pair][old_word]
            if not pair_to_words[pair]:
                del pair_to_words[pair]
    
        word_counts[old_word] -= freq
        if word_counts[old_word] <= 0:
            del word_counts[old_word]
        
        new_word = merge_word(old_word, best_pair)

        word_counts[new_word] += freq

        new_pair_counts = pairs_in_word(new_word)
        for pair, counts in new_pair_counts.items():
            pair_counts[pair] += freq * counts
            pair_to_words[pair][new_word] += 1
        
    return word_counts, pair_counts, pair_to_words

def build_word_counts_for_chunks(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str]
):
    tqdm.write("Building word counts with parallel pretokenization")
    compiled_pat = re.compile(GPT2_PAT)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    
    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
    
    # 预分词，以特殊标记为分隔划分文本
    if special_tokens:
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        pattern =  "|".join(sorted(escaped_tokens, key=len, reverse=True))  # ty:ignore[no-matching-overload]
        segments = [seg for seg in re.split(pattern, chunk_text) if seg]
    else:
        segments = [chunk_text]

    # 计数：预分词字节元组的频次
    word_counts = Counter()
    for segment in segments:
        # 以符号为分隔切割文本
        for match in compiled_pat.finditer(segment): 
            pretoken = match.group(0)
            # 转换为字节元组
            byte_tuple = word_to_byte_tuple(pretoken)
            word_counts[byte_tuple] += 1
    
    return word_counts

def _process_chunks(task):
    input_path, start, end, special_tokens = task
    return build_word_counts_for_chunks(input_path, start, end, special_tokens)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1,
    **kwargs
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 初始化词表
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8") # 特殊标记直接加入词表不做分词
    if len(vocab) > vocab_size:
        raise ValueError(f" vocab_size={vocab_size} 太小，至少要容纳 256 个字节和 {len(special_tokens)} 个 special_tokens.")

    # 并行化处理 word_counts
    with open(input_path, "rb") as f:
        if special_tokens:
            split_token = special_tokens[0].encode("utf-8")
            boundaries = find_chunk_boundaries(f, num_processes, split_token) # b"<|endoftext|>"
        else:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            boundaries = [0, file_size]

    word_counts = Counter()
    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with mp.Pool(processes=num_processes) as pool:
        for partial_counts in tqdm(
            pool.imap_unordered(_process_chunks, tasks),
            total=len(tasks),
            desc="Pretokenizing chunks",
        ):
            word_counts.update(partial_counts)
    
    # 构建索引：pair_counts -> 每个相邻对出现频次 和 pair_to_words -> 每个相邻对出现在哪些词里和出现频次
    pair_counts, pair_to_words = build_pair_index(word_counts)

    # 合并
    tqdm.write("Running merge")
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab) # 还需要合并多少次才能达到 vocab_size

    for _ in tqdm(range(num_merges), desc="BPE merges"):
        if not pair_counts:
            break

        # 选择出现频次最高的 pair 进行合并，频次相同的情况下按字典序选择
        best_pair = max(pair_counts.items(), key=lambda x:(x[1], x[0]))[0]
        merge_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = merge_token # 将新 token 加入词表

        # 更新中间状态：词频、pair 频次、pair 到词的索引
        word_counts, pair_counts, pair_to_words = apply_merge(word_counts, pair_counts, pair_to_words, best_pair)

    return vocab, merges

def save_vocab(vocab: dict[int, bytes], output_path: str | os.PathLike):
    with open(output_path, "wb") as f:
        pickle.dump(vocab, f)

def save_merges(merges: list[tuple[bytes,bytes]], output_path: str | os.PathLike):
    with open(output_path, "wb") as f:
        pickle.dump(merges, f)

def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    num_processes = 28

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes)

    tqdm.write("Saving outputs")
    save_vocab(vocab, "data/tinystories_vocab.pkl")
    save_merges(merges, "data/tinystories_merges.pkl")

if __name__ == "__main__":
    main()
