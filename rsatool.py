# do what the calculator program could not

exp = input("exp: ")
exp = int(exp)

mod = input("mod: ")
mod = int(mod)

while(True):
    num = input("Enter num or q to quit: ")
    
    if not(num.isnumeric()):
        break
    else:
        num = int(num)
        
    result = (num**exp)%mod

    print(result)