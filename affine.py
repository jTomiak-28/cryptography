# affine cipher encryption and decryption

from alphaconvert import toNum, toAlpha
from line_input import custom_input

# encrypt using affine cipher. If gcd(26,a)!=1 decryption is not possible.
# text      plaintext to encrypt, string unless numeric is true, then list of ints
# a, b      (int) encryption parameters on [0,25], together they are the cipher key
# numeric   (Boolean) when true text and output are list of ints on [0,25] representing letters
def affine_encrypt(pt, a, b, numeric=False):
    ct = []
    if not(numeric):
        pt = toNum(pt) # convert to numbers
    for i in pt:
        ct.append( (a*i+b)%26 ) # use affine encrytption formula to find ct
    if not(numeric):
        ct = toAlpha(ct) # convert to string
    return ct

# decrypt using affine cipher. If gcd(26,a)!=1 decryption is not possible.
# text      ciphertext to decrypt, string unless numeric is true (then list of ints)
# a, b      (int) encryption parameters on [0,25], together they are the cipher key
# numeric   (Boolean) when true text and output are list of ints on [0,25] representing letters
def affine_decrypt(ct, a, b, numeric=False):
    a_inv = pow(a,-1,26) # find mod inverse of a, throws ValueError if it doesn't exist
    pt = []
    if not(numeric):
        ct = toNum(ct)
    for i in ct:
        pt.append( (a_inv*(i-b)%26) )
    if not(numeric):
        pt = toAlpha(pt)
    return pt

# interface with user via command line to encrypt/decrypt with affine cipher
def affine_main():
    print("\n" + "--------Affine cipher encryption and decryption--------")
    a = custom_input("a: ", "whole")
    b = custom_input("b: ", "whole")
    text = custom_input("Text: ", "alpha")
    use = custom_input("Encrypt (0) or decrypt (1)? ", "bin")
    if use == 0:
        print(affine_encrypt(text,a,b))
    if use == 1:
        print(affine_decrypt(text,a,b))

if __name__ == "__main__":
    affine_main()