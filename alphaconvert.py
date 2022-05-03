# convert letters to their number in the alphabet (A=0,Z=25) and vice versa

# given a string of lower or uppercase letters, returns list of corresponding integers
def toNum(text):
    outlist = []
    text = text.upper()
    convert_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    outlist = [convert_str.index(char) for char in text if char in convert_str]
    return outlist

# given integer list, returns string of corresponding uppercase letters
def toAlpha(numlist, check=True):
    convert_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    outstring = ""
    for i in numlist:
        if not(i>=0 and i<=25):
            if check:
                raise ValueError("input must be int on [0,25]")
            else:
                continue
        outstring += convert_str[i]
    return outstring