'''#### Tokenizer
    已经有了BPE构建出来的词表，接下来是如何使用这个词表。

    一个完整的tokenizer分为encoder跟decoder两个部分。

        encoder：利用词表将词转换成词元ID。例如：牛->666
        decoder：利用词表将词元ID转换为词。例如：666->牛

    接下来要做的事情就是把输入的词进行拆分和合并，转换成词表中的ID，以及反过来，复原出词
'''
from collections.abc import Iterable
import os

import regex as re
import json

class BPETokenizer:
    """
    字节级 BPE（Byte-Pair Encoding）分词器实现。
    
    该分词器将任意字符串编码为整数 ID 序列，并能将 ID 序列还原。
    它采用字节级处理，确保不会出现未知词（OOV）错误。

        （1）初始化词表、分词规则和合并规则
        （2）将输入句子转换为字节序列（特殊token->分词->BPE）不过这里仅仅针对输入的内容进行BPE
        （3）根据合并规则进行 BPE 合并，得到最终的词元序列，得注意合并的顺序，要按merge的来            
        （4）将词元序列转换为对应的 ID 序列（用的就是词表）
    """

    def __init__(self, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        '''
        初始化 BPE 分词器，只要调用了该类就会自动执行。
            （1）初始化词表、分词规则和合并规则
        '''
        # 1. 先构建反向映射（需要 bytes_to_unicode 函数）
        # 构建反向映射：可见字符 -> 字节值
        # 没有这个的话，后续if pair in self.merges_dict:'永远无解，因为pair是bytes类型，mereges是str类型
        self.unicode_to_byte = {v: k for k, v in self.bytes_to_unicode().items()}
        # 2. 再加载词表和合并规则
        self.vocab = self.load_vocab(vocab_path)
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        self.merges = self.load_merges(merges_path)
        # 3. 构建 merges_dict
        self.merges_dict = {pair: i for i, pair in enumerate(self.merges)}

        # 匹配特殊token，这个与BPE算法中的有些区别，这里需要保留每一句的特殊token，而BPE算法中第一步是去除特殊token
        if special_tokens:
            # 在编码时，特殊 token 和普通文本混合在一起，必须优先匹配最长的特殊 token，防止长 token 被短 token 拆开。例如 <|endoftext|> 必须优先于 <|end|>
            special_tokens = sorted(special_tokens, key=len, reverse=True)

            # 创建正则表达式模式，用于匹配特殊 token
            regex_pattern = '|'.join(re.escape(token) for token in special_tokens)

            self.special_tokens_pattern = re.compile(regex_pattern)
        else:
            self.special_tokens_pattern = None

        # 分词：GPT-2 官方预分词正则表达式。
        # 它的作用是在应用 BPE 合并前，先将文本切分成单词、标点、数字等逻辑块。
        # 这样做是为了防止 BPE 规则跨越单词或标点（例如：防止将 "dog" 的末尾和 "." 合并）。
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # 输入无特殊token的文本，直接进行BPE合并，并且得到对应的ID序列用于后续训练
    def encode_for_only_text(self, text: str) -> list[int]:
        '''
        将输入的文本进行BPE编码，返回对应的词元ID列表。
        '''

        token_ids = []
        # 1. 将输入文本进行预分词，得到初始的词元列表
        tokens = self.gpt2_pat.findall(text)
        # print(f"Tokens after pre-tokenization: {tokens}")

        # 2. 对每个词元进行字节级编码，得到对应的字节序列(可以跟之前的BPE复用的，但是我这重新写了)
        # 'Hello' ---> [b'H', b'e', b'l', b'l', b'o']
        for token in tokens: # 对输入中的每一个预分词单元都进行操作，token表示的就是预分词单元'Hello'
            bytes_token = [
                bytes([b]) for b in token.encode("utf-8") # 循环这里一定要写token
            ]
            # bytes_token = [b'H', b'e', b'l', b'l', b'o']
            # 遍历每一个预分词单元，进行合并，小于1的就不需要合并了
            while len(bytes_token) > 1: # 因为在过程中会合并bytes_token，所以这个while循环到时候会自己跳出
                # 3. 对字节序列进行 BPE 合并，得到最终的词元列表
                best_pair = None
                # 在当前序列的所有相邻对中，寻找合并优先级最高（Rank 最小）的一对，即按照构造merge时添加pair的顺序进行合并
                minist_rank = float('inf') 
                for i in range(len(bytes_token) - 1):
                    # 因为每一个预分词单元都不是很长，所以直接遍历来找这一次要合并的pair就可以了
                    pair = (bytes_token[i], bytes_token[i + 1])
                    if pair in self.merges_dict:
                        rank = self.merges_dict[pair]
                        if rank < minist_rank:
                            minist_rank = rank
                            best_pair = pair

                # 遍历结束这一轮以后发现没有需要合并的，直接跳出while循环，进入下一个预分词
                if best_pair is None:
                    break

                # 下面这种写法其实挺不错的，不过实际中可能更喜欢最后一起处理，而不是走一步看一步
                # ==================================================
                # 对rank优先级最高的pair进行合并
                # new_token = best_pair[0] + best_pair[1]
                # bytes_token[i] = new_token
                # del bytes_token[i + 1]

                # # 重置搜索，重新寻找最佳合并对
                # best_pair = None
                # minist_rank = float('inf')
                # i = 0  # 重置索引，重新开始搜索
                # ==================================================

                new_tokens = [] # 用来存合并后的token的，将直接赋值给bytes_token，最后就是合并后的预分词单元的字节流了



                # 由于一个预分词单元中存在best_pair的可能不止一个地方，所以仍然需要遍历
                # 与前面BPE不同的是，前面的BPE不需要进行完整的列表遍历，只要剩余的组不成pair对就直接跳过了
                # 下面这个不行，需要遍历完整列表，因为会将'new_tokens = []'直接赋值给bytes_token，所以就算无法组成pair对的也要遍历到
                i = 0
                while i < len(bytes_token):
                    # 开始合并
                    if (i < len(bytes_token) - 1) and ((bytes_token[i], bytes_token[i+1]) == best_pair):
                        new_token = best_pair[0] + best_pair[1]
                        new_tokens.append(new_token)
                        i += 2 # 处理完当前pair对以后，直接跳到下一个pair对去审计，到尾巴就等于一轮结束，如果 len(bytes_token) > 1，说明该预分词的合并还没结束
                    else:
                        # 没有可以合并的，就直接加入
                        new_tokens.append(bytes_token[i])
                        i += 1

                # 处理完这个预分词单元，把现在的合并后的状态覆盖掉原来的状态
                bytes_token = new_tokens

            # 处理完某一个预分词单元以后，要将其转换成数字ID
            # 原因：Embedding 层本质上是一个查找表，只能接收整数索引来提取对应的向量
            for byte in bytes_token:
                token_ids.append(self.byte_to_id[byte])
                # ["Hello", " world"] ——--> [1547, 2310, 123, 56, 78, 90]
                # 没有再按预分词单元进行边界的分隔了，因为后续输入到模型中就是一个扁平的序列，不能还有嵌套，如'[[1547, 2310], [123, 56, 78, 90]]'
                # 不可能出现“未知字节组合”。最终 byte_parts 中的每个元素，要么是原始单字节，要么是严格按照 merges 规则合并出来的 token。这些全部都在词表里。

        return token_ids


    # 输入文本，返回对应的词元ID列表，包含特殊token的处理
    def encode(self, text: str) -> list[int]:
        # 输入为空
        if text is None:
            return []

        # 找到输入文本中的特殊token位置，进行特殊处理
        # 因为特殊token由多个字符构成，没办法直接作为文本处理

        if not self.special_tokens_pattern:
            # 没有特殊token，直接进行BPE编码
            return self.encode_for_only_text(text)

        # 有特殊token，进行额外处理
        # Hello<|endoftext|>I'm a student.  --->  list[list[str]]: [['Hello'], ['<|endoftext|>'], ['I'm a student']] ---> list[int]: [123, 56, 35, 78, 90, 456...]
        token_ids = []
        text_start = 0
        
        # “遍历 text 字符串中所有匹配 self.special_tokens_pattern 正则表达式的位置，每次匹配的结果存放到 match 变量中，并执行循环体内的代码。”
        # match.group()：返回匹配到的特殊 token 字符串本身（如 "<|endoftext|>"）。
        # match.start()：匹配项在 text 中的起始索引位置。
        # match.end()：匹配项在 text 中的结束索引位置（不含）

        # 返回一个re.Match 对象的迭代器。一个一个地“懒惰”生成匹配结果，用到一个才产生一个
        for match in self.special_tokens_pattern.finditer(text):
            # 处理特殊 token 之前的普通文本
            if match.start() > text_start:
                normal_text = text[text_start:match.start()]
                token_ids.extend(self.encode_for_only_text(normal_text))
                # 最后输出的是list[int]，每一次循环出来的都是一个list[int]，要把他们合在一起，需要使用extend方法，而不是append方法，因为append最后的输出会是list[list[int]]
                # pre_tokens : [1,2,3,...] self._encode_text_segment: [4,5,6] tokens.extend -> [1,2,3,...,4,5,6]
                # token.append() : [1,2,3,...,[4,5,6]]

            # 处理特殊 token
            special_token = match.group()
            token_ids.append(self.byte_to_id[special_token.encode("utf-8")])

            # 更新 text_start 为当前特殊 token 的结束位置，以便下一次循环处理特殊 token 之前的普通文本
            text_start = match.end()

            # 也许会存在 Hello <|endoftext|> I'm a student <|endoftext|> nice day
            # 所以这个部分仅仅需要处理特殊token之前以及特殊token就可以了，特殊token之后的文本放到最后一起处理

        # 处理特殊 token 之后的普通文本
        if text_start < len(text):
            normal_text = text[text_start:]
            token_ids.extend(self.encode_for_only_text(normal_text))

        return token_ids

    # 输入词元ID列表，返回对应的文本（包含特殊token的处理）
    def decode(self, token_ids: list[int]) -> str:
        # 将词元ID列表转换为对应的字节序列
        byte_seq = b''.join(self.vocab[token_id] for token_id in token_ids)
        # 将字节序列解码为字符串
        text = byte_seq.decode("utf-8", errors="replace")
        return text

    # 用于unicode--->bytes的反向映射
    def bytes_to_unicode(self):
        """
        创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
        这是 GPT-2 源码中的标准做法。
        0: b'\x00', 1: b'\x01' ---> 0: 'Ā', 1: 'ā'
        """
        # 把一个字符（长度为1的字符串）转换成它对应的 Unicode 码点（Code Point），结果是一个整数（默认以十进制显示）
        # 这么写是因为中间有一部分字符json保存不了，不合法
        # 这一步出来的结果全是数字
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
        return dict(zip(bs, cs)) # { int : str }
            
    def load_vocab(self, vocab_path: str) -> dict[str, int]:
        # 加载词表，直接进行反向映射，使得str--->bytes
        with open(vocab_path, 'r', encoding='utf-8') as f:
            json_vocab = json.load(f)

        vocab = {}

        for k,v in json_vocab.items():
            # v 是可见字符组成的字符串，还原为字节序列
            # 'self.unicode_to_byte[ch] for ch in v)'把字符转成对应的数字 
            byte_seq = bytes(self.unicode_to_byte[ch] for ch in v)
            vocab[int(k)] = byte_seq
        
        return vocab

    def load_merges(self, merges_path: str) -> list[tuple[str, str]]:
        # 加载合并规则，直接进行反向映射，使得str--->bytes
        merges = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                # 将每个可见字符字符串还原为字节序列
                # 'self.unicode_to_byte[ch] for ch in parts[0]'-----------> 把字符转成对应的数字 
                # Ġ t--------------------> 32 116
                # self.unicode_to_byte { str : int }
                b1 = bytes(self.unicode_to_byte[ch] for ch in parts[0])
                b2 = bytes(self.unicode_to_byte[ch] for ch in parts[1])
                merges.append((b1, b2)) # (b' ', b't')
        return merges

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        流式编码器（生成器版本），用于处理超大文本或流式数据。

        当你有一个很大的文件（比如几十 GB）无法一次性读入内存时，
        可以逐行（或分块）读取，然后通过此方法逐个产出 token ID，
        避免内存爆炸。

        参数:
            iterable: 任何可迭代的字符串对象，比如：
                - 文件句柄（with open(...) as f）
                - 列表 / 生成器 / 迭代器

        返回:
            一个生成器（generator），每次 yield 一个整数 ID。
            你可以直接用 for 循环遍历它，也可以送入模型或写入磁盘。

        用法示例:
            with open("huge_corpus.txt", "r", encoding="utf-8") as f:
                for token_id in tokenizer.encode_iterable(f):
                    # 每次只处理一个 ID，内存占用恒定
                    process(token_id)
        """
        for chunk in iterable:
            # 对当前块（通常是一行文本）执行完整的 BPE 编码
            # 得到该块的 ID 列表，然后逐个 yield 出去
            # 注意：yield from 会将列表中的每个元素逐个产出，
            # 而不是把整个列表作为一个元素产出
            yield from self.encode(chunk)

    def main(self):
        # 测试 encode_for_only_text 方法
        
        test_text = "Hello, world! This is a test."
        self.encode_for_only_text(test_text)
        


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "../data/TinyStoriesV2-GPT4/")
    tokenizer = BPETokenizer(vocab_path= path + "vocab.json", merges_path= path + "merges.txt", special_tokens=["<|endoftext|>", "<|pad|>"])
    tokenizer.main()
