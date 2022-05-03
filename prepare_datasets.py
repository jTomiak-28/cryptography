# build datasets for cryptography plaintext recognition project

import pandas as pd
import numpy as np
from hillcipher import hill_encrypt, hill_decrypt
from affine import affine_decrypt, affine_encrypt
from alphaconvert import toNum, toAlpha

trainfile = "data/twitter_data.csv"
testfile = "data/twitter_testdata.csv"

trainframe = pd.read_csv(trainfile, header=None, encoding="ISO-8859-1", usecols=[5])
testframe = pd.read_csv(testfile, header=None, encoding="ISO-8859-1", usecols=[5])

# lists to eventually become final dataframes
final_train = []
final_test = []

# simplify each tweet to a string of letters and encode each letter as a number 0-25
for i in range(len(trainframe)):
    text = trainframe.iloc[i,0]
    text = text.upper()
    # convert text (only using letters) to list of ints 0-25
    convert_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rowdata = [convert_str.index(char) for char in text if char in convert_str]
    rowdata.insert(0,1) # insert class label as first entry
    final_train.append(rowdata)

#now same for test data
for i in range(len(testframe)):
    text = testframe.iloc[i,0]
    text = ''.join(filter(str.isalpha, text))
    rowdata = toNum(text)
    rowdata.insert(0,1)
    final_test.append(rowdata)
    

# generate negative samples by encrypting text and decrypting with wrong key
negative_train = []
for i in range(len(final_train)):
    text = final_train[i].copy() # get ith element from positive samples
    del text[0] # remove label
    if (i%2)==0: # for half use Hill cipher
        while True: # generate false key (not the inverse of the real key)
            key = np.random.randint(0,26,(2,2))
            badkey = np.random.randint(0,26,(2,2))
            try:
                # errors if mod inverse of either key matrix DNE
                det = np.linalg.det(key)
                pow(int(round(det)%26), -1, 26)
                det = np.linalg.det(badkey)
                pow(int(round(det)%26), -1, 26)
            except:
                continue
                
            # only reaches here if both inv keys exist
            if not(np.array_equal(key,badkey)): # keys must be different
                ct = hill_encrypt(text, key, 2, True)
                badpt = hill_decrypt(ct, badkey, 2, True) # failed decryption
                badpt.insert(0,0) # add class label as first entry
                negative_train.append(badpt)
                break
    
    else: # for other half use affine cipher
        while True:
            a = np.random.randint(0,26)
            b = np.random.randint(0,26)
            bad_a = np.random.randint(0,26)
            bad_b = np.random.randint(0,26)
            try:
                pow(a, -1, 26) # if a has no mod 26 inverse, try with a new a
                pow(bad_a, -1, 26)
            except:
                continue
                
            # only reaches here if both a and bad_a are valid
            if a != bad_a or b != bad_b:
                ct = affine_encrypt(text,a,b,True)
                badpt = affine_decrypt(text,bad_a,bad_b,True)
                badpt.insert(0,0) # add class label as first entry
                negative_train.append(badpt)
                break

# now repeat for test
negative_test = []
for i in range(len(final_test)):
    text = final_test[i].copy() # get ith element from positive samples
    del text[0] # remove label
    if (i%2)==0: # for half use Hill cipher
        while True: # generate false key (not the inverse of the real key)
            key = np.random.randint(0,26,(2,2))
            badkey = np.random.randint(0,26,(2,2))
            try:
                # errors if mod inverse of either key matrix DNE
                det = np.linalg.det(key)
                pow(int(round(det)%26), -1, 26)
                det = np.linalg.det(badkey)
                pow(int(round(det)%26), -1, 26)
            except:
                continue
                
            # only reaches here if both inv keys exist
            if not(np.array_equal(key,badkey)): # keys must be different
                ct = hill_encrypt(text, key, 2, True)
                badpt = hill_decrypt(ct, badkey, 2, True) # failed decryption
                badpt.insert(0,0) # add class label as first entry
                negative_test.append(badpt)
                break
    
    else: # for other half use affine cipher
        while True:
            a = np.random.randint(0,26)
            b = np.random.randint(0,26)
            bad_a = np.random.randint(0,26)
            bad_b = np.random.randint(0,26)
            try:
                pow(a, -1, 26) # if a has no mod 26 inverse, try with a new a
                pow(bad_a, -1, 26)
            except:
                continue
                
            # only reaches here if both a and bad_a are valid
            if a != bad_a or b != bad_b:
                ct = affine_encrypt(text,a,b,True)
                badpt = affine_decrypt(text,bad_a,bad_b,True)
                badpt.insert(0,0) # add class label as first entry
                negative_test.append(badpt)
                break

# add negative samples to pre-dataframe nested lists
for row in negative_train:
    final_train.append(row)
for row in negative_test:
    final_test.append(row)

# pad/clip samples to max length
max_length = 20

# train
for row in final_train:
    while len(row) < (max_length + 1): # +1 for class label
        row.append(26) # pad value is 26
    while len(row) > (max_length + 1):
        del row[-1]

# test
for row in final_test:
    while len(row) < (max_length + 1): # +1 for class label
        row.append(26) # pad value is 26
    while len(row) > (max_length + 1):
        del row[-1]

# make dataframes and shuffle the negatives and positives together
final_trainframe = pd.DataFrame(final_train)
final_trainframe = final_trainframe.sample(frac=1)

final_testframe = pd.DataFrame(final_test)
final_testframe = final_testframe.sample(frac=1)

final_trainframe.to_csv("data/train.csv", index=False, header=False)
final_testframe.to_csv("data/test.csv", index=False, header=False)