import pytest


def add_numbers(num1, num2):
    return num1 + num2


def test_add_numbers():
    assert add_numbers(1, 2) == 3
    assert add_numbers(-1, 1) == 0
    assert add_numbers(-1, -1) == -2


def main():
    num1 = float(input("Enter the first number: "))
    num2 = float(input("Enter the second number: "))

    sum_result = add_numbers(num1, num2)
    print(f"The sum of {num1} and {num2} is: {sum_result}")


if __name__ == "__main__":
    main()
