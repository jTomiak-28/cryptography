# Cryptography
*Projects relating to my 2022 cryptography class*

This excellent 2022 cryptography class frustrated me in that we were asked to perform lots of cryptographic functions (encryption, decryption, and cracking) for a variety of ciphers by HAND. This quickly became tedious and I began writing programs to automate much of the work (with instructor permission of course). By the end of the semester I had enough code written that I decided to do a coding-based, scientific final paper rather than the typical historical/humanities style final.

So, the bulk of the code here is for my end-of-semester paper on plaintext recognition- that is, determining whether a given string of text is a correctly decrypted ciphertext or a garbled mess made from attempting to decrypt with the wrong key.  In actual cryptography jargon this is the concept of fitness, and the plaintext recognition programs are called fitness functions.  It's pretty neat stuff.

In my project I tried a few different machine learning approaches to plaintext recognition and reported the results.  I built a 3.2 M sample dataset (from Sentiment140 Twitter dataset) where half the instances are the original messages and half are the messages encrypted and incorrectly decrypted with affine and 2x2 Hill ciphers.  The datasets were too large to include here but they can be easily recreated with these programs.  I then trained a logit, SVM, and CNN to classify plaintext vs. garble, with the CNN doing far better at 95%.  Then when that had some probems in application I developed a word lookup method that does much better.  By the end I developed two programs capable of automatically cryptanalyzing those two cipher types with the word lookup method (hillbreaker2.py and affinebreaker.py).  I did include the paper in this repo cuz why not.

Besides that there are some random programs mainly coding out encryption, decryption, and cryptanalysis for a couple of ciphers like we learned in class.
