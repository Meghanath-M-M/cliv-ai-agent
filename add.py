#!/usr/bin/env python

def add(a, b):
    return a + b

def main():
    num1 = int(input('Enter the first number: '))
    num2 = int(input('Enter the second number: '))
    result = add(num1, num2)
    print(f'The sum of {num1} and {num2} is {result}')
if __name__ == '__main__':
    main()