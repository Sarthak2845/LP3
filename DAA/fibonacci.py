def fibonacci_iterative(n):
    a = 0
    b = 1
    print("Fibonacci Series:")
    for i in range(n):
        print(a, end=" ")
        a, b = b, a + b
n = int(input("Enter the number of terms: "))
fibonacci_iterative(n)
# recusive function to calculate fibonacci
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
n = int(input("Enter the number of terms: "))
print("Fibonacci Series:")
for i in range(n):
    print(fibonacci_recursive(i), end=" ")