import math, sys
import sympy as sp

# ax^3 + bx^2 +cx + d = 0, a, b, c, d must be integers
a = 1
b = 4
c = 17
d = 26

p = []
q = []

def factorize(num):
    factors = []
    n = abs(num)
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors) + sorted([-i for i in factors], reverse=True)
    
p = factorize(d)
q = factorize(a)

print(f"p: {p}")
print(f"q: {q}")

pq = []

for i in p:
    for j in q:
        if sp.Rational(i, j) not in pq:
            pq.append(sp.Rational(i, j))

print(f"p over q: {pq}")

def isDivisible(k):
    val = (a * (k ** 3)) + (b * (k ** 2)) + (c * k) + d
    #print(k, val)
    if val == 0:
        return True
    return False

def synthDiv(k):
    # global a, b, c, d
    na = a
    nb = a * k + b
    nc = a * (k ** 2) + b * k + c
    nd = a * (k ** 3) + b * (k ** 2) + c * k + d
    
    # a, b, c, d = na, nb, nc, nd
    return na, nb, nc, nd

realFactor = None
for possibleFactor in pq:
    if isDivisible(possibleFactor):
        realFactor = possibleFactor
        break
if realFactor == None:
    print('no rational solutions found')
    sys.exit()

(na, nb, nc, nd) = synthDiv(realFactor)
#print(na, nb, nc, nd)

def quadraticFormula(a, b, c):
    ra = sp.Rational(a)
    rb = sp.Rational(b)
    rc = sp.Rational(c)

    disc = (rb ** 2) - (4 * ra * rc)
    #print(disc)

    sqr1 = sp.sqrt(disc)
    sqr2 = -sp.sqrt(disc)
    #print(sqr1)
    #print(sqr2)

    top1 = (-rb) + sqr1
    top2 = (-rb) + sqr2

    root1 = top1/(2 * ra)
    root2 = top2/(2 * ra)

    return root1, root2

otherRoots = quadraticFormula(na, nb, nc)
print(f'Rational Factor: {realFactor}')
print(f'Other Roots: {otherRoots}')
