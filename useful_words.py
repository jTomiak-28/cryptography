# use plaintext recognition dataset to build lists of words from nltk corpus that
# work well in plaintext recognition- keep words that appear in real plaintexts but not
# in ciphertexts

from nltk.corpus import words
import pandas as pd
from alphaconvert import toAlpha


# large list of english words from nltk
nltk_words = words.words()

# bring in dataset (just first 10,000 instances)
pt_data = pd.read_csv("data/train.csv", header=None, nrows=10000)


# convert numeric pandas dataset to text list
positive_list = []
negative_list = []
for i in range(len(pt_data)):
    text = toAlpha(pt_data.iloc[i, 1:], check=False)
    text = text.lower() # nltk dataset is lowercase
    if pt_data.iloc[i, 0]: # use class labels to sort instances
        positive_list.append(text)
    else:
        negative_list.append(text)


# build final list of useful words in positive instances
useful_words = []
for word in nltk_words:
    pos_count = 0
    neg_count = 0
    for pos in positive_list:
        if word in pos:
            pos_count += 1
    for neg in negative_list:
        if word in neg:
            neg_count += 1
    if pos_count > 1 and neg_count < 2:
        useful_words.append(word)

print("len(useful_words): ", len(useful_words))

filename = 'data/useful_words.txt'

words_str = '\n'.join(useful_words)
file = open(filename, 'w', encoding='utf-8')
file.write(words_str)
file.close()
