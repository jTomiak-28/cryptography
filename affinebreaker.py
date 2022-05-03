# automatic affine cipher cryptanalysis using brute force attack with ML to check results

import torch
from text_recog_cnn import CNN
from line_input import custom_input
from alphaconvert import toNum, toAlpha
from affine import affine_decrypt
from dictionary_check import count_words, load_words

# given affine ct, tries every possible key and returns list of top results as scored
# by ML algo trained to distinguish garble from real plaintext
def break_affine(ct, model_file="models/final_cnn.pth", use_words=True):
    top_results = []
    if use_words: # prep list for word recog if needed
        wordlist = load_words()
    else:
        model = torch.load(model_file) # load ML model
    for a in range(26):
        try: # check if a has mod inverse
            pow(a,-1,26)
        except:
            continue
        for b in range(26):
            trial_pt = affine_decrypt(ct, a, b, True)
            if use_words:
                score = count_words(trial_pt[:100], wordlist, numeric=True)
            else:
                score = model.eval_text(trial_pt) # get legibility score from model
            trial_pt = toAlpha(trial_pt)
            results_tuple = (trial_pt, a, b, score)
            
            if len(top_results) < 10:
                top_results.append(results_tuple)
            else:
                for i in range(len(top_results)): # maintain top 10 list
                    if score > top_results[i][3]:
                        top_results.insert(i,results_tuple)
                        break
                while len(top_results) > 10:
                    del top_results[-1]
    
    return top_results


# interface with user via command line to break affine cipher
def affinebreaker_main():
    print("\n" + "--------Affine cipher cryptanalysis--------" + "\n")
    ciphertext = custom_input("Enter ciphertext: ", "alpha")
    ct = toNum(ciphertext)
    results_list = break_affine(ct, use_words=False) # add use_words=False argument to use CNN (worse)
    for result in results_list:
        print(result)

if __name__ == "__main__":
    affinebreaker_main()