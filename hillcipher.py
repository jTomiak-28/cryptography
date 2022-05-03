# encrypt or decrypt with the Hill cipher using command line interaction in main()

import numpy as np
from modmatrix import modinv
from alphaconvert import toNum, toAlpha


# function hill_encrypt(): encrypt text with hill cipher, returning string.
# Alternatively, decrypts when passed the inverse of key.
# text (str)            plaintext to encrypt
# key (np.ndarray)      square key matrix
# dim (int)             size of matrix (and text chunks)
# numeric (Boolean)     when true text and output are list of ints on [0,25] representing letters
def hill_encrypt(text, key, dim, numeric=False):
    ct = []
    if not(numeric):
        pt = toNum(text)
    else:
        pt = text
    while(len(pt)%dim != 0): # ensure text is divisible into blocks for multiplication
        pt.append(0)
    for i in range(0, len(pt), dim): # for each block of plaintext
        ptblock = pt[i:i+dim] # get block
        ptblock = np.asarray(ptblock)
        result = np.matmul(ptblock, key) # encrypt with matrix multiplication
        result = result.tolist()
        for j in result:
            ct.append(int(j)%26)
    if not(numeric):
        ct = toAlpha(ct) # convert to string
    return ct


# decrypt text with hill cipher, returning string
# text (str)            ciphertext to decrypt
# key (np.ndarray)      square key matrix
# dim (int)             size of matrix (and text chunks)
# numeric (Boolean)     when true text and output are list of ints on [0,25] representing letters
def hill_decrypt(text, key, dim, numeric=False):
    inv_key = modinv(key)
    return(hill_encrypt(text, inv_key, dim, numeric)) # decryption is same as encryption with inverted key


# interact with user via command line to encrypt or decrypt input using Hill cipher
def hill_main():
    # get key matrix from user
    print("\n" + "--------Hill cipher encryption and decryption--------")
    print("Note: to decrypt given the inverse of the key,")
    print("give the inverse and choose encrypt.\n")
    while True:
        dim = input("Enter dimension of key matrix: ")
        if dim.isnumeric():
            if int(dim) > 0:
                dim = int(dim)
                break
        print("Invalid input- enter a positive number")
    key = np.empty([dim,dim])
    for i in range(dim):
        row = input("Enter row " + str(i+1) + " (separated by spaces): ").split()
        for j in range(len(row)):
            row[j] = int(row[j])
        key[i] = row
    
    # get input text
    while True:
        text = input("Enter message to encrypt/decrypt: ")
        # remove spaces
        text = text.replace(" ", "")
        if text.isalpha():
            break
        print("Invalid input- enter letters only")

    # encrypt or decrypt
    while True:
        use = input("Encrypt (0) or decrypt (1)? ")
        if use=='0':
            print(hill_encrypt(text, key, dim))
            break
        elif use=='1':
            print(hill_decrypt(text, key, dim))
            break
        print("Invalid input- enter 0 or 1")
        
        return
    
if __name__ == "__main__":
    hill_main()
