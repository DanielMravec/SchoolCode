import math, sys

print('This is an imaginary number calculator')

def main():
    def isInt(input: str):
        try:
            int(input)
        except:
            return False
        return True
    def operationInput():
        operation = input('What operation would you like to use (+ - * / sqrt)? ')
        operations = ['+', '-', '*', '/', 'sqrt']
        if not operation.lower() in operations:
            print('Try again')
            main()
            sys.exit()
        return operation.lower()
    def aInput():
        a = input('What is the \'a\' value in a + bi? ')
        if not isInt(a):
            print('Try again')
            main()
            sys.exit()
        return int(a)
    def bInput():
        b = input('What is the \'b\' value in a + bi? ')
        if not isInt(b):
            print('Try again')
            main()
            sys.exit()
        return int(b)
    def add(imag1: tuple, imag2: tuple):
        end = (imag1[0] + imag2[0], imag1[1] + imag2[1])
        return end
    def sub(imag1: tuple, imag2: tuple):
        end = (imag1[0] - imag2[0], imag1[1] - imag2[1])
        return end
    def mul(imag1: tuple, imag2: tuple):
        front = imag1[0] * imag2[0]
        outer = imag1[0] * imag2[1]
        inner = imag1[1] * imag2[0]
        last = imag1[1] * imag2[1]
        end = (front - last, outer + inner)
        return end
    def div(imag1: tuple, imag2: tuple):
        opposite = (imag2[0], -imag2[1])
        numerator = mul(imag1, opposite)
        denomerator = (imag2[0] ** 2) + (imag2[1] ** 2)
        end = (numerator[0] / denomerator, numerator[1] / denomerator)
        return end
    def newsqrt(imag: tuple):
        def X_YToDir_Dist(cord: tuple):
            x = cord[0]
            y = cord[1]
            dist = math.sqrt((x**2)+(y**2))
            #print('distance is {}'.format(dist))
            ang = math.asin(y/dist)
            #print('angle is {}'.format(ang))
            return (ang, dist)
        def Dir_DistToX_Y(cord: tuple):
            ang = cord[0]
            dist = cord[1]
            x = dist * math.cos(ang)
            #print('x is {}'.format(x))
            y = dist * math.sin(ang)
            #print('y is {}'.format(y))
            return (x, y)
        cord = X_YToDir_Dist(imag)
        angle = cord[0] / 2
        dist = math.sqrt(cord[1])
        newCord = (angle, dist)
        end = (Dir_DistToX_Y(newCord), (-Dir_DistToX_Y(newCord)[0], -Dir_DistToX_Y(newCord)[1]))
        return end
    operation = operationInput()
    a = aInput()
    b = bInput()
    imag1 = (a, b)
    if operation != 'sqrt':
        a2 = aInput()
        b2 = bInput()
        imag2 = (a2, b2)
    if operation == '+':
        ans = add(imag1, imag2)
    elif operation == '-':
        ans = sub(imag1, imag2)
    elif operation == '*':
        ans = mul(imag1, imag2)
    elif operation == '/':
        ans = div(imag1, imag2)
    else:
        ans = newsqrt(imag1)
        print('One answer is {} + {}i'.format(ans[0][0], ans[0][1]))
        print('The other answer is {} + {}i'.format(ans[1][0], ans[1][1]))
        sys.exit()
    print('The answer is {} + {}i'.format(ans[0], ans[1]))

main()
        
    
    
    




