def add(a, b):
    print("Result:", a + b)
def subtract(a, b):
    print("Result:", a - b)
def multiply(a, b):
    print("Result:", a * b)
def divide(a, b):
    if b == 0:
        print("Cannot divide by zero.")
    else:
        print("Result:", a / b)
a = float(input("Enter first number: "))
b = float(input("Enter second number: "))
operator = input("Enter operator (+, -, *, /): ")
if operator == "+":
    add(a, b)
elif operator == "-":
    subtract(a, b)
elif operator == "*":
    multiply(a, b)
elif operator == "/":
    divide(a, b)
else:
    print("Invalid operator!")