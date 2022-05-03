# use euclidean algo to find gcd or coefficients s and t


# calculate gcd using euclidean algorithm
def gcd(a,b):
    r = a % b
    if r != 0:
        b = gcd(b,r)
    # once r = 0 b is the gcd
    return b


# Returns coefficients s and t such that s*a + t*b = gcd(a,b) (useful to find mod inv)
# Uses recursive equations I found to automate solving for s and t as we
# learned in class. The recursion equation uses the previous two coefficients and
# the current quotient to find the current coefficients of a and b that form the
# current remainder, like so:
#
# [s(n),t(n)] = [s(n-2) - q(n)*s(n-1), t(n-2) - q(n)*t(n-1)]
#
# such that s(n)*a + t(n)*b = r(n)
# where q(n) is nth quotient, s(n),t(n) are nth coefs, r(n) is nth remainder
#
# I also found that for this to work, the initial coefficients are:
# [s(-1),t(-1)] = [1,0]
# [s(0),t(0)] = [0,1]
# 
def find_coef(a, b, older_coef=[1,0], newer_coef=[0,1]):
    # (above) if first iteration (no coef passed), set intial coefficients
    # [s(n-2), t(n-2)] = [1,0]
    # [s(n-1), t(n-1)] = [0,1]
    q = int(a/b)
    r = a % b
    s = older_coef[0] - q * newer_coef[0]
    t = older_coef[1] - q * newer_coef[1]
    if r != 0:
        newer_coef = find_coef(b, r, newer_coef.copy(), [s,t])
    return newer_coef
    

# use user input to either find gcd or s and t
def euclid_main():
    print("\n--------Euclidean algorithm tool--------")
    # let user choose use case
    while True:
        use = input("Enter 1 to find gcd(a,b) or 2 to find\n"
                    + "[s,t] such that s*a + t*b = gcd(a,b): ")
        if use == '1':
            use = 1
            break
        if use == '2':
            use = 2
            break
    
    while True:
        a = input("a: ")
        if a.isnumeric():
            a = int(a)
            break
        print("Invalid input- enter a positive integer")
    
    while True:
        b = input("b: ")
        if b.isnumeric():
            b = int(b)
            break
        print("Invalid input- enter a positive integer")
    
    if use == 1:
        print( gcd(a,b) )
    elif use == 2:
        print( find_coef(a,b) )


if __name__ == "__main__":
    euclid_main()