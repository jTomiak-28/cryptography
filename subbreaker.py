# Brute force attack w/ frequency analysis to break monoalphabetic substitution ciphers

# path for cmd cd access
# C:\Users\Tomiakfamily\Documents\Govschewlwork\2021-2022\Cryptography\Python

import torch
from text_recog_cnn import CNN
from line_input import custom_input
from alphaconvert import toNum, toAlpha
from subcipher import sub_decrypt
from dictionary_check import count_words, load_words

# Attempts to break given ct. Generates possible keys by searching the space of possible
# keys near the 'ideal' key- the key for which the observed letter frequencies match
# those expected for English language- then tries these keys and checks their results via
# ML model for identifying correct plaintext. Returns list of possible plaintexts,
# corresponding keys, and their score from the ML model on a worst 0 to best 1 scale.
# ct        (list) ciphertext to decrypt as list of ints on [0,25] representing letters
# num_swaps (int) higher gives larger search
def break_sub(ct, num_swaps=10, model_file="models/final_cnn.pth", use_words=True):
    ct_freq = [i for i in range(26)]
    ct_freq.sort(reverse=True, key=ct.count) # make list of letters sorted by descending freq in ct
    
    if use_words: # prepare evaluation tool (wordlist or ML model)
        wordlist = load_words()
    else:
        model = torch.load(model_file) # load ML model
    swaps = []
    top_results = []
    while len(swaps) < num_swaps:
        swaps = iter_swaps(swaps)
        trial_key = key_swap(ct_freq, swaps)
        trial_key = sort_to_eng(trial_key)
        trial_pt = sub_decrypt(ct, trial_key, numeric=True, keycheck=False)
        if use_words:
            score = count_words(trial_pt[:40], wordlist, numeric=True)
        else:
            score = model.eval_text(model_input)
        results_tuple = (toAlpha(trial_pt), toAlpha(trial_key), score)
        
        if len(top_results) < 10:
            top_results.append(results_tuple)
        else:
            for i in range(len(top_results)): # maintain top 10 list
                if score > top_results[i][2]:
                    top_results.insert(i,results_tuple)
                    del top_results[-1]
                    break
    
    # once finished return top 10 results
    return top_results


# Given list of ints representing the 26 letters sorted descending by their frequency
# in some text, returns the letters mapped to their position by english frequencies.
# eg: the first (most common) element in the input will be moved to index 4 (letter E)
def sort_to_eng(freq_key):
    # list of english letters as ints on 0-25 sorted ascending by expected frequencies
    eng_freq = [4,19,0,14,8,13,18,17,7,3,11,20,2,12,5,24,22,6,15,1,21,10,23,16,9,25]
    sorted_key = []
    for i in range(26):
        sorted_key.append(freq_key[eng_freq.index(i)])
    return sorted_key


# Helper function for search of key space. Iterates given list of swaps to eventually
# produce every unique key through swaps between adjacent elements of the 'ideal' key.
# Returns next valid swaps list to pass to key_swap(). Pass empty list to start and cut
# off search based on len(swaps). Criteria to ensure search of unique swap combinations
# (new value is rightmost so elements to the left have already been checked):
#   - New value is not equal to the value to its left (the two cancel out if so)
#   - New value is greater than the value to its left, or is smaller but consecutive
#       However, if consecutive, the new value is not the same as any previous value
#       (Prevents redundant swap combinations like [0,2] and [2,0])
def iter_swaps(swaps):
    if len(swaps) == 0: # starting search with empty list
        return [0]
    if len(swaps) == 1: # just one swap in list
        if swaps[0] == 24: # on to len(swaps) = 2
            swaps = [0,0]
        else:
            swaps[0] += 1
            return swaps
    while True: # increment through swaps until unique swaps is found
        if all([swap==24 for swap in swaps]): # when all variables have reached max
            swaps = [0 for i in range(len(swaps))] # move to higher len(swaps)
            swaps = iter_swaps(swaps)
            swaps.append(0)
        else:
            swaps[-1] += 1
        if swaps[-1] == 25: # when the right value is maxed, iterate inner values
            del swaps[-1]   # and reset right value to 0
            swaps = iter_swaps(swaps)
            swaps.append(0)
        # next check swaps with the above criteria for unique search
        if (swaps[-1] != swaps[-2]):
            if swaps[-1] > swaps[-2]:
                return swaps
            elif swaps[-1] == swaps[-2]-1:
                if all([swaps[-1] != swap for swap in swaps[:-1]]):
                    return swaps


# Helper function for search of key space. Swaps elements of base_key at indices
# specified in swaps with the element to their right in the order appearing in swaps
# base_key      (int list) 26 unique ints on [0,26] generated in break_sub()
# swaps         (int list) swap positions in order
# returns base_key with given swaps made
def key_swap(base_key, swaps):
    new_key = base_key.copy()
    for swap in swaps:
        new_key.insert(swap+1, new_key.pop(swap))
    return new_key


def subbreaker_main():
    print("\n" + "-----Monoalphabetic substitution cipher cryptanalysis-----" + "\n")
    ciphertext = custom_input("Enter ciphertext: ", "alpha")
    ct = toNum(ciphertext)
    num_swaps = custom_input("Search size (number of swaps, default 10): ", "whole")
    results_list = break_sub(ct, num_swaps)
    for result in results_list:
        print(result)
    

if __name__ == "__main__":
    subbreaker_main()