## 2. Byte-pair encoding (BPE) Tokenizer
## 2.1 the unicode standard
print(repr(chr(0)))
print("A",chr(0),"A")
print("this is a test" + chr(0) + "string")
# chr(0) give a space?
# print(ord('牛'))
# print(ord('傻'))
# print(ord('s'))

# UTF-8
test_string = "困"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded) # print 16 x
print(list(utf8_encoded)) # print 10 x
# first number in binary 
# [229, 155, 176] 
# \xe5 229 11100101
# \x9b 155 10011011
# \xb0 176 10110000
# first place binary‘s prefix tells you how many "next" bits are the same "word"
# start with 0 means only cuurent for rest count how many 1's in the prefix
# 10 is means this bit belonging to some word

# choose utf8 since it each token is only range from 0 - 255
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# print(re.findall(PAT, "some text that i'll pre-tokenize"))
## find all will insert all result into the list and store in memory at once.  !!!!
# When using it in your code, however, you should use re.finditer to avoid storing the pre-tokenized 
# words as you construct your mapping from pre-tokens to their counts.
# re.finditer only store the current value in order to save memory



# (a) Consider a GPT-2 XL-sized model using our assignment architecture, which has the following 
# configuration:
# vocab_size:  50,257
# context_length:  1,024
# num_layers:  48
# d_model:  1,600
# num_heads:  25
# d_ff:  4,288 (the nearest multiple of 64 to 8
# 3 × 1, 600)
# Suppose we constructed our model using this configuration. How many trainable parameters 
# would our model have? Assuming each parameter is represented using single-precision floating 
# point, how much memory is required to just load this model?
# Deliverable: A one-to-two sentence response.
embedding_trainable = 50257 * 1600
# 1,600 RMSNrom -> KQV + projection 4*1600*1600  -> 1,600 RMSNrom -> SwiGLU 3 * d_ff * d_model
each_tranformer_block = 1600 * 2 + (1600*1600)*4  + 3 * 4288 * 1600
