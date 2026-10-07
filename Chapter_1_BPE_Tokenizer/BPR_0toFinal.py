##### 完整的写一遍BPE算法，以及补充保存的函数思路
#（1）文件路径：../data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train.txt
#（2）最终得到的词表和合并顺序会给tokenizer使用

# 初始化 vocab
def init_vocab():
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    return vocab

import regex as re

# --------------- 去除特殊token ------------------
def del_special_token(segment:str, special_tokens:list[str])-> list[str]:
    if len(special_tokens) < 1:
        return [segment]

    # for special_token in special_tokens:
        # 将特殊token直接合起来，方便re.split()用
        # "|".join(re.escape(special_token)) 会对转义后的字符串逐字符用 | 连接
        # split_rule = "|".join(re.escape(special_token))
    special_token_pattern = "|".join(re.escape(x) for x in special_tokens)
    # print(special_token_pattern) # \<\|endoftext\|\>|\<\|startoftext\|\>|\<\|pad\|\>|\<\|unk\|\>|\<\|mask\|\>
    split_text = re.split(f'{special_token_pattern}', segment)
    
    # 根据特殊token分词以后，再把句子合并起来
    # 过滤空字符串和特殊 token（实际上 parts 中已无特殊 token）
    final_segment = [token.strip() for token in split_text if token not in special_tokens and token.strip()]
    
    return final_segment # 返回一个list，因为去掉特殊token以后，一句话可能变成n句话，得用列表存

# --------------- 预分词 segment_list:list[str]直接传预分词的结果, 并统计预分词单元词频 ----------------
from collections import Counter
def pre_split(segment_list:list[str], pattern: str | None = None) -> tuple[list[str], list[int]]:
    """
    对文本片段列表进行预分词，并统计词频。

    参数:
        segment_list: 去除特殊 token 后的普通文本片段列表。
        pattern: 自定义预分词正则表达式（字符串）。若为 None，使用 GPT-2 默认正则。

    返回:
        tuple[list[str], list[int]]:
            - word_list: 预分词后的扁平列表，如 ["Hello", " world", "!"]
            - count_list: 与 word_list 一一对应的词频列表
    """
    if pattern is None:
        compiled_re = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    else:
        compiled_re = re.compile(pattern)

    count_counter = Counter()

    for segment in segment_list:
        splited_segment = compiled_re.findall(segment)
        count_counter.update(splited_segment)

    word_list = list(count_counter.keys())
    count_list = list(count_counter.values())

    return word_list, count_list
        
    # append 是把整个列表作为一个元素塞进去，结果变成嵌套列表。
    # 结果：result = [['我很喜欢LLM', '，', '目前正在手搓一个LLM'], ['从最基础的BPE开始', '，', '希望我能成功', '。', 'Hello', ' Word', '!']]
    # extend 是把列表里的每个元素单独拿出来添加，结果保持扁平列表。
    # 结果：['我很喜欢LLM', '，', '目前正在手搓一个LLM', '从最基础的BPE开始', '，', '希望我能成功', '。', 'Hello', ' Word', '!']

# ==================== 预分词结束，开始BPE算法 ======================

# ---------- BPE算法第一步：将每一个预分词单元转换成字节序列 ----------------
def word_trans_to_token(word_list:list[str]) -> list[list[bytes]]:
    token_list = []
    for word in word_list:
        token_flow = [
            # i.encode('utf-8') for i in word # 错误写法，实现不了字节流BPE
            bytes([b])
            for b in word.encode("utf-8") 
            # word指的是预分词单元，例如'Hello'，可不是'H', 'e', 'l', 'l', 'o'
            # 写循环的时候这个部分要注意一下
        ]
        token_list.append(token_flow)
    return token_list

# ---------- BPE算法第二步：得到预分词的字节序列，统计pair对和对应的pair词频 ---------------
from collections import defaultdict

# word_list 保存唯一预分词单元，
# count_list 保存每个预分词单元对应的频率，
# 两者一一对应，避免重复统计 pair 频率。
def build_pair_stats_and_indices(token_list:list[list[bytes]], count_list:list[int]) -> tuple[defaultdict, defaultdict]:
    # stats 统计所有 pair 的全局频率
    # indices 建立 pair -> word_id 的倒排索引
    # token_list= [[b'H', b'e', b'l', b'l', b'o'],...]]
    stats  = defaultdict(int) # 用来统计词频
    indices = defaultdict(set) # 倒排索引
    for idx, token in enumerate(token_list): # 一个一个预分词单元来
        # 经过 word_trans_to_token()，word=[b'H', b'e', b'l', b'l', b'o']
        freq = count_list[idx]
        for i in range(len(token) -1) : # 预分词单元中一个一个字符来
            pair = (token[i], token[i+1])
            stats[pair] += freq
            indices[pair].add(idx)
        
    return stats,indices

def pre_split_from_file(input_path: str, special_tokens: list[str], pattern: str | None = None) -> tuple[list[str], list[int]]:
    """
    从文件流式读取，去除特殊 token，预分词并统计词频。
    
    参数:
        input_path: 语料文件路径
        special_tokens: 特殊 token 列表（如 ["<|endoftext|>"]）
        pattern: 自定义预分词正则（若为 None，使用 GPT-2 默认）
    
    返回:
        tuple[list[str], list[int]]: (word_list, count_list)，两者一一对应
    """
    if pattern is None:
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    compiled_re = re.compile(pattern)
    count_counter = Counter()
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            # 去除特殊 token
            segments = del_special_token(line, special_tokens)
            for seg in segments:
                tokens = compiled_re.findall(seg)
                count_counter.update(tokens)
    
    word_list = list(count_counter.keys())
    count_list = list(count_counter.values())
    return word_list, count_list

# 统计好pair对以及对应频率了，接下来就该开始BPE最核心的部分
def merge_token(vocab: dict[int, bytes],
                merges: list[tuple[bytes, bytes]], 
                token_list:list[list[bytes]], 
                count_list: list[int],
                stats:defaultdict ,indices:defaultdict):
    """
    执行一步 BPE 合并：找最佳 pair，更新 token_list、stats 和 indices。
    """
    
    if not stats:
        return

    # 找最佳 pair
    best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]
    new_token = best_pair[0] + best_pair[1]

    new_id = len(vocab)
    vocab[new_id] = new_token

    merges.append(best_pair) # 保存 merge 顺序，Tokenizer 按相同顺序即可重建词表
    
    affected_indices = list(indices[best_pair])
    for idx in affected_indices:
        tokens = token_list[idx]          # 当前单词的字节列表
        freq = count_list[idx]    # 当前单词的频率
        
        i = 0
        while i < len(tokens) - 1: # merge 以后token 数量发生变化所以不能写：for i in range(len(tokens) - 1):
            if tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
               
                # 删除旧 pair
                if i > 0:
                    left_pair = (tokens[i-1], tokens[i])
                    if left_pair in stats:
                        stats[left_pair] -= freq
                        if stats[left_pair] <= 0:
                            del stats[left_pair]
                    if left_pair in indices:
                        indices[left_pair].discard(idx)
                        if not indices[left_pair]:
                            del indices[left_pair]
                
                if i + 2 < len(tokens):
                    right_pair = (tokens[i+1], tokens[i+2])
                    if right_pair in stats:
                        stats[right_pair] -= freq
                        if stats[right_pair] <= 0:
                            del stats[right_pair]
                    if right_pair in indices:
                        indices[right_pair].discard(idx)
                        if not indices[right_pair]:
                            del indices[right_pair]
                
                # 执行合并
                tokens[i:i+2] = [new_token]
                
                # 添加新 pair
                if i > 0:
                    new_left = (tokens[i-1], new_token)
                    stats[new_left] += freq
                    indices[new_left].add(idx)
                if i + 1 < len(tokens):
                    new_right = (new_token, tokens[i+1])
                    stats[new_right] += freq
                    indices[new_right].add(idx)

                # 合并后，索引 i 指向的是新 Token。
                # i 不需要移动（i+=1），因为我们刚刚修改了 word[i] 并且删除了 word[i+1]。
                # 下一轮循环会检查新的 (word[i], word[i+1])，即 (new_token, old_word[i+2])
                # 这可以处理像 A A A A -> X A A这样的情况，正确地更新新的邻居对
                
            else:
                i += 1

    if best_pair in stats:
        del stats[best_pair]

    if best_pair in indices:
        del indices[best_pair]

def bytes_to_unicode():
    """
    创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
    这是 GPT-2 源码中的标准做法。
    0: b'\x00', 1: b'\x01' ---> 0: 'Ā', 1: 'ā'
    """
    # 把一个字符（长度为1的字符串）转换成它对应的 Unicode 码点（Code Point），结果是一个整数（默认以十进制显示）
    # 这么写是因为中间有一部分字符json保存不了，不合法
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    # 补充原文缺少的字符，确保所有 256 个字节都有对应的 Unicode 字符
    # 0 对应的值是 256 → chr(256) → 'Ā'（可见字符，JSON 安全）
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            # 就是这一步'cs.append(256 + n)'，把“坏字节”（如 0、1、32）映射到一个全新的、JSON 安全的码点（如 256、257、288），
            # 从而让这些原本在 JSON 中非法或不可见的字节，有了一个“合法且可见的替身”。
            n += 1
    # chr(n) 函数会把数字转换成它对应的Unicode字符：
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

import json
import os

def save_vocab_and_merges(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], out_dir:str):
    """
    # ==================== 注意点 =====================
    # 保存方式：
    #   1. 使用 bytes_to_unicode() 将每个字节映射为可见的 Unicode 字符。
    #   2. 对于 token 中的每个字节，用映射后的字符替换，并拼接成字符串。
    #   3. vocab 保存为 JSON 文件，键为 ID，值为拼接后的字符串。
    #   4. merges 保存为 TXT 文件，每行两个 token，用空格分隔。
    #   5. 加载时，通过反向映射（即 unicode -> byte）恢复原始字节序列。
    #   这种方式确保：
    #      - JSON 安全：所有字符都是可见且合法的 JSON 字符串。
    #      - 可读：英文字母保持原样，控制字符用替代符号表示。
    #      - 无损还原：映射是双射，可完全恢复原始字节。
    # ============================================================
    """
    ''' Json更通用，一些web api都这样
    JSON：存的是数据本身（自带结构、类型和解析规则），程序无需额外说明就能直接读取其内容。

    TXT：存的是纯文字（无结构、无类型），程序必须由你亲自编写解析规则，才能理解其含义。
    '''

    # 保存vocab词表为json文件

    # 先将原本0-255的字节映射为可见的Unicode字符，确保JSON安全
    # byte_to_unicode = dict(sorted(bytes_to_unicode().items()))
    byte_to_unicode = bytes_to_unicode()

    # 利用映射表格进行转换，将字节流转换为可见字符
    json_vocab = {
        k: "".join(byte_to_unicode[b] for b in v)
        for k, v in vocab.items()
    }
    # print(byte_to_unicode)
    # print(json_vocab) 

    # 用的unicode字符，所以必须用utf-8编码保存
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, indent=4) #  indent=4 表示用 4 个空格缩进
        # 为了保证生成的 JSON 文件在任何操作系统、任何编辑器下都不乱码，Python 默认会把非 ASCII 字符（如 Ā、Ġ、中文）转义成 \u 开头的 Unicode 码点。
        # \u0100 和 Ā 只是同一个字符的两种写法，在程序运行时完全等价，可以直接互换使用
    
    # 合并规则保存
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 同样转换 p1 和 p2
            s1 = "".join(byte_to_unicode[b] for b in p1)
            s2 = "".join(byte_to_unicode[b] for b in p2)
            f.write(f"{s1} {s2}\n")

    

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "../data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train.txt")
    output_dir = os.path.join(script_dir, "../data/TinyStoriesV2-GPT4/")
    os.makedirs(output_dir, exist_ok=True)

    vocab_size = 10000 # 表示合并次数，数量可以不包含特殊的token。
    special_tokens = ["<|endoftext|>"]

    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    # 1. 读取语料
    # with open(input_path, "r", encoding="utf-8") as f:
    #     full_text = f.read()

    # 2. 去除特殊 token
    # segments = del_special_token(full_text, special_tokens)

    # 3. 预分词并统计词频
    # word_list, count_list = pre_split(segments)

    # 流式预分词并统计词频
    word_list, count_list = pre_split_from_file(input_path, special_tokens)

    # 4. 转换为字节序列
    token_list = word_trans_to_token(word_list)

    # 5. 构建初始 pair 统计和倒排索引
    stats, indices = build_pair_stats_and_indices(token_list, count_list)

    # 6. 初始化词表和合并列表
    vocab = init_vocab()
    merges = []

    # 7. 计算需要合并的次数，其实也不用， 超了Python 不会报错，因为字典没有硬性长度限制。
    # 但是为了调试方便等，还是需要限制一下的
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges <= 0:
        raise ValueError(f"vocab_size 必须大于 {256 + len(special_tokens)}")

    # 8. 迭代合并
    print(f"开始训练 BPE 分词器 (目标词表大小: {vocab_size})...")
    for step in range(num_merges):
        if not stats:
            print(f"第 {step} 步：无可合并的 pair，提前终止")
            break
        merge_token(vocab, merges, token_list, count_list, stats, indices)
        if (step + 1) % 1000 == 0:
            print(f"已完成 {step+1} 次合并，当前词表大小: {len(vocab)}")

    # print(vocab) # {0: b'\x00', 1: b'\x01', 2: b'\x02', 3: b'\x03', 4: b'\x04',..., 256: b'\xe6\x88'}
    # 词表需要保存的是能够看懂的token，而不是字节流，所以需要把字节流转换成可读的字符串，例如：‘牛’
        
    # print(merges)

    # --- 添加特殊 token ---
    # 这里加入一个特殊toke正好10000 = vocab_size。后续要扩大需要修改vocab_size参数，或者在这里添加更多特殊token。
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")


    # 9. 保存结果
    save_vocab_and_merges(vocab, merges, output_dir)
    print(f"训练完成！最终词表大小: {len(vocab)}，合并规则数: {len(merges)}")


if __name__ == "__main__":
    main()


        