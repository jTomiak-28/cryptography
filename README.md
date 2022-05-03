# Cryptography
*Projects relating to my 2022 cryptography class*

There are a number of unrelated programs on here, but the bulk of the code is for my end-of-semester paper on plaintext recognition- that is, determining whether a given string of text is a correctly decrypted ciphertext or a garbled mess made from attempting to decrypt with the wrong key.  In actual cryptography jargon this is the concept of fitness, and the plaintext recognition programs are called fitness functions.  It's pretty neat stuff.

In my project I tried a few different machine learning approaches to plaintext recognition and reported the results.  I built a 3.2 M sample dataset (from Sentiment140 Twitter dataset) where half the instances are the original messages and half are the messages encrypted and incorrectly decrypted with affine and 2x2 Hill ciphers.  The data was saved with all but letters stripped and letters converted to integer encodings like A --> 0, B --> with 26 padding to length 20 where needed (train.csv and test.csv).  I then trained a logit, SVM, and CNN to classify plaintext vs. garble, with the CNN doing far better at 95%.  Then when that had some probems in application I developed a word lookup method that does much better.  By the end I developed two programs capable of automatically cryptanalyzing those two cipher types with the word lookup method (hillbreaker2.py and affinebreaker.py).  I did include the paper in this repo cuz why not.

Besides that there are some random programs mainly coding out encryption, decryption, and cryptanalysis for a couple of ciphers like we learned in class.
