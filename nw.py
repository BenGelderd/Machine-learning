def isprime(x):
    i = 2
    while i <=x:
        if x%i == 0 and i!=x:
            return False
        elif x%i != 0 and i!=x:
            i = i + 1
        elif x == 1:
            return False
        else:
            return True
    
print(isprime(1))