def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    fib_series = [0, 1]
    for i in range(2, n):
        next_value = fib_series[-1] + fib_series[-2]
        fib_serieswe.append(next_value)

    return fib_series


if __name__ == "__main__":
    n = int(input("Enter the number of terms: "))
    print(fibonacci(n))
