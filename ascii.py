# convert numbers to ASCII codes

from line_input import custom_input

# return list of ASCII values in order for each character of the input string
def get_ascii(message):
    codes = []
    for i in message:
        codes.append(ord(i))
    return codes

# return plaintext given list of ASCII values
def get_plain(codes):
    chars = []
    for i in codes:
        chars.append(chr(i))
    return ''.join(chars)

def ascii_main():
    use = custom_input("Encode (0) or decode (1) ASCII codes? ", "bin")
    if use == 0:
        message = custom_input("Enter message to encode: ", "alpha")
        print(get_ascii(message)) 
    elif use == 1:
        codes = custom_input("Enter numbers to convert to ASCII: ", "list")
        print(get_plain(codes))

if __name__ == "__main__":
    ascii_main()
