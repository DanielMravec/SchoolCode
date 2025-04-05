import math, sys

leg1=input('leg1 for right triangle give now ')
leg2=input('leg2 for right triangle give now ')

try:
    int(leg1)
except:
    sys.exit()

leg1 = int(leg1)

try:
    int(leg2)
except:
    sys.exit()

leg2 = int(leg2)

hyp = math.sqrt((leg1**2)+(leg2**2))
area = leg1 * leg2 / 2

print('Hypotenuse of triangle is {}. Area of the triangle is {}'.format(hyp, area)
