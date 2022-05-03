# automatic hill cipher cryptanalysis using brute force attack with ML to check results

import torch
import numpy as np
from text_recog_cnn import CNN
from line_input import custom_input
from alphaconvert import toNum, toAlpha
from hillcipher import hill_decrypt
from dictionary_check import count_words, load_words

# given hill ct, tries every possible 2x2 key matrix and returns list of top 10 results
# as scored by an ML algo trained to distinguish garble from real plaintext
def break_hill2(ct, model_file="models/final_cnn.pth", use_words=True):
    top_results = []
    model = torch.load(model_file)
    if use_words: # prep list for word recog if needed
        wordlist = load_words()
    else:
        model = torch.load(model_file) # load ML model
    for a in range(26):
        for b in range(26):
            for c in range(26):
                for d in range(26):
                    key = np.array([[a,b],[c,d]])
                    try: # if key matrix is not invertible, hill_decrypt raises ValueError
                        trial_pt = hill_decrypt(ct, key, 2, True)
                    except ValueError:
                        continue
                    
                    if use_words:
                        score = count_words(trial_pt[:100], wordlist, numeric=True)
                    else:
                        score = model.eval_text(trial_pt) # get legibility score from model
                    trial_pt = toAlpha(trial_pt)
                    results_tuple = (trial_pt, key, score)
                    
                    # if the list is empty or score is 1.0, this run makes it in the list
                    if len(top_results) < 10:
                        top_results.append(results_tuple)
                    else:
                        for i in range(len(top_results)): # maintain top 10 list
                            if score > top_results[i][2]:
                                top_results.insert(i,results_tuple)
                                del top_results[-1]
                                break
    
    return top_results


# interface with user via command line to break affine cipher
def hillbreaker2_main():
    print("\n" + "--------Automatic Hill cipher cryptanalysis--------" + "\n")
    ciphertext = custom_input("Enter ciphertext: ", "alpha")
    ct = toNum(ciphertext)
    results_list = break_hill2(ct, use_words=True) # pass use_words=False argument to use CNN (worse)
    for result in results_list:
        print(result)

if __name__ == "__main__":
    hillbreaker2_main()