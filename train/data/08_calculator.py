# simple calculator
def calculate(a, op, b):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        if b == 0:
            return "Cannot divide by zero"
        return a / b
    else:
        return "Unknown operator"


a = int(input("First number: "))
op = input("Operator (+, -, *, /): ")
b = int(input("Second number: "))
print(calculate(a, op, b))
