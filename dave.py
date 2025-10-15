def a_s(x):
    i = 1
    val = 0
    while i < x:
        if x%i == 0:
            val = val + i
            i = i + 1
        else:
            i = i + 1
    return val

print(a_s(12))