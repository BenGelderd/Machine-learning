def cmult(w, z):
    lst1 = [(w[0] * z[0]) - (w[1] * z[1]), (w[0] * z[1]) + (w[1] * z[0])]
    return lst1

def csetmult(lst):
    z1 = lst[0]
    P = z1
    x = len(lst)
    print(P)
    for i in range(x - 1):
        Pr = cmult(P, lst[i + 1])
        P = Pr
        print(P)
    return P

lst = [[1, 2], [1, 4], [1, 5]]
print(csetmult(lst))
