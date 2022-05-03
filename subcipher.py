# monoalphabetic substitution cipher encryption and decryption given key alphabet

import random
from line_input import custom_input
from alphaconvert import toNum, toAlpha

# returns pt encrypted with substitution cipher method using given key alphabet
# pt        (str or int list) plaintext to encrypt
# key       (str or int list) 26 unique letters (or numbers on [0,25])
# numeric   (Boolean) if true inputs and outputs are int lists
# keycheck  (Boolean) whether to check key alphabet to ensure it is usable
def sub_encrypt(pt, key, numeric=False, keycheck=True):
    ct = []
    if not(numeric):
        pt = toNum(pt)
        key = toNum(key)
    if keycheck:
        if len(key) != 26:
            raise ValueError("key must have exactly 26 unique letters")
        for char in key:
            if key.count(char) != 1:
                raise ValueError("key must have exactly 26 unique letters")
    for i in pt:
        ct.append(key[i])
    if not(numeric):
        ct = toAlpha(ct)
    return ct

# returns ct decrypted with substitution cipher method using given key alphabet
# ct        (str or int list) ciphertext to decrypt
# key       (str or int list) 26 unique letters (or numbers on [0,25])
# numeric   (Boolean) if true inputs and outputs are int lists
# keycheck  (Boolean) whether to check key alphabet to ensure it is usable
def sub_decrypt(ct, key, numeric=False, keycheck=True):
    pt = []
    if not(numeric):
        ct = toNum(ct)
        key = toNum(key)
    if keycheck:
        if len(key) != 26:
            raise ValueError("key must have exactly 26 unique letters")
        for char in key:
            if key.count(char) != 1:
                raise ValueError("key must have exactly 26 unique letters")
    for i in ct:
        pt.append(key.index(i))
    if not(numeric):
        pt = toAlpha(pt)
    return pt

# generates random key alphabet, or given keyword removes repeated letters then adds remaining letters in alphabetical order
# keyword       (str) if using simple substitution keyword method, keyword to make key with
# numeric       if true changes input (keyword) and return types to lists of int
def make_keyalphabet(keyword=None, numeric=False):
    if keyword == None:
        key = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
        random.shuffle(key)
    else:
        if not(numeric): 
            keyword = toNum(keyword)
        key = list(dict.fromkeys(keyword)) # remove duplicates from keyword
        for i in range(26):
            if not(i in key): # add remaining alphabet letters in order
                key.append(i)
    if not(numeric):
        key = toAlpha(key)
    return key


def sub_main():
    print("\n" + "--------Substitution cipher encryption and decryption--------")
    use = custom_input("Encrypt (0) or decrypt (1)? ", "bin")
    key = custom_input("Enter full 26 letter key alphabet, keyword to make the\n" +
    "rest of the alphabet with, or A to generate random key:\n", "alpha")
    if key == "A":
        key = make_keyalphabet()
        print("Random key alphabet in use: ", key)
    elif len(key) < 26:
        key = make_keyalphabet(keyword=key)
        print("Key alphabet from keyword in use: ", key)
    
    if use == 0:
        pt = custom_input("Enter message to encrypt: ", "alpha")
        print("ciphertext: ", sub_encrypt(pt, key))
    elif use == 1:
        ct = custom_input("Enter message to decrypt: ", "alpha")
        print("plainttext: ", sub_decrypt(ct, key))


if __name__ == "__main__":
    sub_main()