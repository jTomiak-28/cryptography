# cryptanalyze a hill cipher given a crib and the size of the key matrix

import numpy as np
from modmatrix import modinv
from hillcipher import hill_encrypt, hill_decrypt
from alphaconvert import toNum, toAlpha

# cryptanalyze ciphertext given a crib and the size of the key matrix. Assumes even
# division into blocks of length dim.
# ct (str)      ciphertext
# crib (str)    crib (must be start of message)
# dim (int)     size of key matrix
def break_hill(ciphertext, cribtext, dim):
    # convert strings to integers
    ct = toNum(ciphertext)
    crib = toNum(cribtext)
    # break each text sequence into blocks
    ct_blocks = []
    crib_blocks = []
    key = np.empty([dim,dim])
    for i in range(0, len(ct), dim):
        ct_blocks.append(ct[i:i+dim])
    for i in range(0, len(crib), dim):
        crib_blocks.append(crib[i:i+dim])
    # try combinations of crib blocks to invert crib matrix to find key matrix
    found = False # loop breaker
    for i in range(len(crib_blocks)):
        for j in range(len(crib_blocks)):
            if i==j: # choose different blocks in the crib
                continue
            crib_mat = np.array([crib_blocks[i],crib_blocks[j]])
            try: # try inverting crib matrix
                crib_inv = modinv(crib_mat)
            except ValueError:
                continue
            # if inverse crib matrix found
            found = True
            ct_mat = np.array([ct_blocks[i],ct_blocks[j]])
            # multiply ct and crib matrices in mod26 to find key matrix
            key = np.matmul(crib_inv, ct_mat)
            key %= 26
            break
        if found:
            break
    # use found key to decrypt ct
    return key, hill_decrypt(ciphertext, key, dim)
    
# interact with user via command line to encrypt or decrypt input using Hill cipher
def hillbreaker_main():
    # get input ciphertext and crib from user
    print("\n" + "--------Hill cipher cryptanalysis--------" + "\n")
    while True:
        ct = input("Enter ciphertext: ")
        ct = ct.replace(" ", "")
        if ct.isalpha():
            break
        print("Invalid input- enter letters only")
    while True:
        crib = input("Enter crib (must start at the beginning of the message): ")
        crib = crib.replace(" ", "")
        if crib.isalpha():
            break
        print("Invalid input- enter letters only")
        
    # get size of key matrix from user
    while True:
        dim = input("Enter dimension of key matrix: ")
        if dim.isnumeric():
            if int(dim) > 0:
                dim = int(dim)
                break
        print("Invalid input- enter a positive number")

    # cryptanalyze
    key, pt = break_hill(ct, crib, dim)
    print("key:\n", key)
    print("pt: ", pt)
    
    return

if __name__ == "__main__":
    hillbreaker_main()