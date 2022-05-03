# routines for taking user input through the command line

import numpy as np

# prints prompt (str) and loops until user enters valid input of specified type.
# in_type has 8 valid values, function returns ValueError if it's anything else:
# "str"     (default) whatever the user enters, no checking (returns str)
# "bin"     0 or 1 (returns int)
# "whole"   positive integer or 0 (returns int)
# "int"     integer, includes negatives (returns int)
# "float"   includes decimals and negatives (returns float)
# "alpha"   string with alphabetic letters only (returns str)
# "list"    list of whole numbers (returns list)
# "matrix"  matrix of any dimension (returns np array)
def custom_input(prompt, in_type="str"):
    if not(in_type in ["str","bin","whole","int","float","alpha","list","matrix"]):
        raise ValueError("Invalid value of in_type")
    while True:
        if in_type != "matrix":
            response = input(prompt)
        else:
            print(prompt)
        
        if in_type == "str":
            return response
        
        elif in_type == "bin":
            if response == '0' or response == '1':
                return int(response)
            else:
                print("Invalid input- enter 0 or 1")
                
        elif in_type == "whole":
            if response.isnumeric():
                return int(response)
            else:
                print("Invalid input- enter a whole number")
                
        elif in_type == "int": # can have one negative sign
            if response.replace('-','',1).isnumeric():
                return int(response)
            else:
                print("Invalid input- enter an integer")
        
        elif in_type == "float": # can have one negative sign and one decimal
            if response.replace('-','',1).replace('.','',1).isnumeric():
                return float(response)
            else:
                print("Invalid input- enter a decimal number")
        
        elif in_type == "alpha":
            if response.isalpha():
                return response
            else:
                print("Invalid input- enter letters only")
        
        elif in_type == "list":
            try:
                response_list = response.split()
                response_list = list(map(int, response_list))
                return response_list
            except:
                print("Invalid input- enter whole numbers separated by spaces")
            
        elif in_type == "matrix":
            dim = custom_input("Enter number of rows of matrix: ", in_type="whole")
            if dim == 0:
                print("Invalid input- enter a positive number of rows")
                continue
            matrix = []
            for i in range(dim):
                matrix.append(custom_input("Row " + str(i+1) + ": ", in_type="list"))
            max_len = max([len(i) for i in matrix]) # check lengths of rows and append with 0 to make them even if necessary
            for i in matrix:
                while len(i) != max_len:
                    i.append(0)
            return np.asarray(matrix)

# Testing:
# print(custom_input("Hola: ", "bin"))