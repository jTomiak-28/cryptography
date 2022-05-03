# use list of first 100 primes to perform prime factorization

from math import sqrt

# list of first 100 primes
primes_list = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]

# check first 100 primes and return the smallest that evenly divides input
def one_factor(num):
    factor = num
    for prime in primes_list:
        if prime > num:
            break
        if (num % prime) == 0:
            factor = prime
            break
    return factor

# return list of prime factors of int num (len 1 list with just the input if prime)
def factor(num):
    factors = []
    while True:
        factor = num
        for prime in primes_list: # check if any prime yields even division
            if prime > num:
                break
            if (num % prime) == 0:
                factor = prime
                break
        factors.append(factor) # add factors to list
        if factors[-1] == num: # repeat until no factor is found
            break
        num = int(num/factor)
    return factors.copy()


# factor using fermat factorization method
def fermat(num):
    x = int(sqrt(num))
    while(True):
        x += 1
        y = sqrt(x**2 - num)
        if (y.is_integer()):
            break
    return [int(x+y), int(x-y)]
    

def factor_main():

    use_fermat = False
    
    while True:
        num = input("Number to factor: ")
        if num.isnumeric():
            num = int(num)
            break
        if num == 'fermat':
            use_fermat = True
            print("Factorization set to fermat method")
            continue
        print("Invalid input")
    
    factors = factor(num)
    
    if use_fermat:
        factors = fermat(num)
    
    if len(factors) == 1:
        print("Prime")
    else:
        print("Factor list: ", factors)

if __name__ == "__main__":
    factor_main()