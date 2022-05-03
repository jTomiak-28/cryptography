# count words in text sequence with list of common words from some guy on github
# https://gist.github.com/deekayen/4148741

from alphaconvert import toAlpha
from nltk.corpus import words

# given str input, returns number of words from nltk words list found in str
def count_words(text, word_list, numeric=False):
    count = 0
    if numeric:
        text = toAlpha(text)
    text = text.lower()
    for word in word_list:
        if word in text and len(word) != 1:
            count += 1
    return count


# return list of 1000 common words from that link up there ^
def load_words():
    if True:
        wordfile = open("data/useful_words.txt", 'r', encoding='utf-8')
        wordlist = wordfile.read()
        wordfile.close()
        wordlist = wordlist.split()
    else:
        wordlist = words.words()
    return wordlist