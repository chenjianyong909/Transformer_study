import json
import os
from collections import Counter, defaultdict
import regex as re

# ---------- 1. 你的完整 train_bpe 函数（基于之前优化版） ----------
def train_bpe(corpus, vocab_size, special_tokens=None):
    """训练 BPE，返回 vocab 和 merges。corpus 为字符串列表。"""
    if special_tokens is None:
        special_tokens = []
    
    # 初始化基础词表
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)
    
    # 预分词（使用GPT-2正则）
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    raw_counts = Counter()
    for text in corpus:
        words = gpt2_pat.findall(text)
        for w in words:
            raw_counts[tuple(bytes([b]) for b in w.encode("utf-8"))] += 1
    
    words_list = [list(k) for k in raw_counts.keys()]
    counts_list = list(raw_counts.values())
    
    stats = defaultdict(int)
    indices = defaultdict(set)
    for idx, word in enumerate(words_list):
        freq = counts_list[idx]
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            stats[pair] += freq
            indices[pair].add(idx)
    
    merges = []
    for step in range(num_merges):
        if not stats:
            break
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
        if stats[best_pair] <= 0:
            break
        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        # 更新词表
        vocab[len(vocab)] = new_token
        
        affected_words = list(indices[best_pair])
        for idx in affected_words:
            word = words_list[idx]
            freq = counts_list[idx]
            i = 0
            while i < len(word)-1:
                if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    # 删除旧pair
                    if i > 0:
                        left = (word[i-1], word[i])
                        stats[left] -= freq
                        if stats[left] <= 0:
                            del stats[left]
                        if left in indices:
                            indices[left].discard(idx)
                            if not indices[left]:
                                del indices[left]
                    if i+2 < len(word):
                        right = (word[i+1], word[i+2])
                        stats[right] -= freq
                        if stats[right] <= 0:
                            del stats[right]
                        if right in indices:
                            indices[right].discard(idx)
                            if not indices[right]:
                                del indices[right]
                    # 合并
                    word[i:i+2] = [new_token]
                    # 添加新pair
                    if i > 0:
                        new_left = (word[i-1], new_token)
                        stats[new_left] += freq
                        indices[new_left].add(idx)
                    if i+1 < len(word):
                        new_right = (new_token, word[i+1])
                        stats[new_right] += freq
                        indices[new_right].add(idx)
                    i += 1
                i += 1
        # 清理best_pair
        if best_pair in indices:
            del indices[best_pair]
        if best_pair in stats:
            del stats[best_pair]
    
    # 加入特殊token（强制放在词表末尾）
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    
    return vocab, merges

# ---------- 2. 保存函数（将vocab和merges转为可JSON序列化的格式） ----------
def save_vocab_and_merges(vocab, merges, vocab_path, merges_path):
    """将vocab和merges保存为JSON文件。vocab中的bytes转为hex字符串。"""
    # vocab: int -> bytes, 转为 int -> hex字符串
    # JSON 不支持 bytes 类型，直接序列化会报错。v.hex()将其转换成16进制保存
    vocab_hex = {str(k): v.hex() for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_hex, f, ensure_ascii=False, indent=2)
    
    # merges: list of tuple(bytes, bytes) -> 转为 [["hex1", "hex2"], ...]
    merges_hex = [[a.hex(), b.hex()] for a, b in merges]
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_hex, f, ensure_ascii=False, indent=2)

# ---------- 3. 加载函数（供tokenizer使用） ----------
def load_vocab_and_merges(vocab_path, merges_path):
    """从JSON文件加载vocab和merges，还原为原始格式。"""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_hex = json.load(f)
    vocab = {int(k): bytes.fromhex(v) for k, v in vocab_hex.items()}
    
    with open(merges_path, "r", encoding="utf-8") as f:
        merges_hex = json.load(f)
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_hex]
    
    return vocab, merges


# ---------- 4. 主程序：使用示例语料训练并保存 ----------
if __name__ == "__main__":
    # 示例语料（可替换为你自己的大语料）
    corpus = [
        "Hello world!",
        "Hello hello world",
        "world wide web",
        "1234567890",
        "Python programming language",
        "BPE tokenization example",
        "你好世界，欢迎使用BPE！",
        "Machine learning is fun."
    ]
    
    # 定义特殊token（必须与后续使用一致）
    special_tokens = ["<|endoftext|>", "<|unk|>", "<|pad|>"]
    
    # 训练 BPE，词表大小设为 300（可根据需要调整）
    vocab, merges = train_bpe(corpus, vocab_size=666, special_tokens=special_tokens)
    
    # 保存到当前目录
    save_vocab_and_merges(vocab, merges, "../Chapter_1_BPE_Tokenizer/vocab/vocab.json", "../Chapter_1_BPE_Tokenizer/vocab/merges.json")
    
    print(f"训练完成！词表大小: {len(vocab)}，合并规则数: {len(merges)}")
    print("已保存 vocab.json 和 merges.json")