import pandas as pd
import numpy as np
import jieba
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pickle
zh = []
en = []
data_dir = '/mnt/c/Users/yutan/Desktop/Data/UM/UM-Corpus/data/Bilingual/News/Bi-News.txt'
num_samples = 6400
with open(data_dir, 'r' ) as f:
    raw = f.read()

for idx, sentence in enumerate(raw.split('\n')):
    if(sentence == ''):
        continue
    if(idx % 2 == 0):
        en.append(sentence)
    else:
        zh.append(sentence)

assert(len(en) == len(zh))


en_tokenized, en_dict, en_counter = [], {'': 0}, 1
zh_tokenized, zh_dict, zh_counter = [], {'': 0}, 1
skipped, last = [], 0
for idx in tqdm(range(len(en))):
    en_tmp = word_tokenize(en[idx])
    if(len(en_tmp) <= 50):
        for i, word in enumerate(en_tmp):
            if(word not in en_dict):
                en_dict[word] = en_counter
                en_counter += 1
            en_tmp[i] = en_dict[word]
        en_tokenized.append(en_tmp)
        tmp= list(jieba.cut(zh[idx]))
        for i, word in enumerate(tmp):
            if(word not in zh_dict):
                zh_dict[word] = zh_counter
                zh_counter += 1
            tmp[i] = zh_dict[word]
        zh_tokenized.append(tmp)
    if(len(en_tokenized)==num_samples):
        break

np.save('data/en' + str(num_samples) + '.npy', np.array(en_tokenized, dtype = object))
with open('data/en' + str(num_samples) + '_dict.pkl', 'wb') as f:
    pickle.dump(en_dict, f)

np.save('data/zh' + str(num_samples) + '.npy', np.array(zh_tokenized, dtype = object))
with open('data/zh' + str(num_samples) + '_dict.pkl', 'wb') as f:
    pickle.dump(zh_dict, f)
# zh = []
# en = []
# data_dir = '/mnt/c/Users/yutan/Desktop/Data/UM/UM-Corpus/data/Bilingual/News/Bi-News.txt'
# num_samples = 6400
# with open(data_dir, 'r' ) as f:
#     raw = f.read()

# for idx, sentence in enumerate(raw.split('\n')):
#     if(sentence == ''):
#         continue
#     if(idx % 2 == 0):
#         en.append(sentence)
#     else:
#         zh.append(sentence)

# assert(len(en) == len(zh))

# en = en[:num_samples]
# en_tokenized, en_dict, en_counter = [], {}, 1

# for idx, sentence in tqdm(enumerate(en), desc = 'en tokenize'):
#     tmp = word_tokenize(sentence)
#     for idx, word in enumerate(tmp):
#         if(word not in en_dict):
#             en_dict[word] = en_counter
#             en_counter += 1
#         tmp[idx] = en_dict[word]
#     en_tokenized.append(tmp)

# np.save('data/en' + str(num_samples) + '.npy', np.array(en_tokenized, dtype = object))
# with open('data/en' + str(num_samples) + '_dict.pkl', 'wb') as f:
#     pickle.dump(en_dict, f)

# zh = zh[:num_samples]
# zh_tokenized, zh_dict, zh_counter = [], {}, 1
# for idx, sentence in tqdm(enumerate(zh), desc = 'zh tokenize'):
#     tmp= list(jieba.cut(sentence))
#     for idx, word in enumerate(tmp):
#         if(word not in zh_dict):
#             zh_dict[word] = zh_counter
#             zh_counter += 1
#         tmp[idx] = zh_dict[word]
#     zh_tokenized.append(tmp)

# np.save('data/zh' + str(num_samples) + '.npy', np.array(zh_tokenized, dtype = object))
# with open('data/zh' + str(num_samples) + '_dict.pkl', 'wb') as f:
#     pickle.dump(zh_dict, f)
# en = np.load('data/en.npy', allow_pickle = True)
# zh = np.load('data/zh.npy', allow_pickle = True)

# en_dict = []
# zh_dict = []
# for tokens in tqdm(en, desc = 'en dict'):
#     for token in tokens:
#         if(token not in en_dict):
#             en_dict.append(token)
# np.save('data/en1600_dict.npy', en_dict)

# for tokens in tqdm(zh, desc = 'zh dict'):
#     for token in tokens:
#         if(token not in zh_dict):
#             zh_dict.append(token)

# np.save('data/zh1600_dict.npy', zh_dict)