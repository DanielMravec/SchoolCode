import sys
sys.set_int_max_str_digits(2147483647)
num = ''
for i in range(2147483647):
    num += '9'
    print(int(num))
print(int(num))
