
# Functions, Methods, and Packages


# Functions
# Functions are the most basic way of packaging code for reuse.
# Functions are defined using the def keyword.

# def function_name(parameters):
#     """docstring"""
#     function_suite
#     return [expression]

# abs() is a built-in function in Python
num1 = 5
num2 = -5
print(abs(num1))
print(abs(num2))

if abs(num1) == abs(num2):
    print("The absolute values of num1 and num2 are equal")

# help() is a built-in function in Python
help(abs)
import math
help(math)

list = [1, 2, 3, 4, 5]
print(max(list))
print(min(list))
x = 5.6
print(round(x))

import math
print(math.floor(x))
print(math.ceil(x))

int1 = '5'
print(int(int1))
print(float(int1))
str1 = 5.6
print(str(str1))

# sum() and round() functions

# round(number, ndigits)
round(5.678, 2)

# sum(iterable, start)
tuple = (1, 2, 3, 4, 5)
print(sum(tuple))

list = [1, 2, 3, 4, 5]
print(sum(list))

# defining a function
# def function_name(parameters):
#     """docstring"""   # optional
#     function_suite
#     return [expression]   # optional

def add(x, y):  # function definition
    """This function adds two numbers"""
    return x + y

print(add(4, 5))  # function call

def haha():
    print("haha")
haha()

def bitcoint_to_usd(btc):
    amount = btc * 527
    print(amount)
bitcoint_to_usd(3.85)

# Methods
# Methods are functions that belong to objects.
# Methods are called using the dot operator.

word = "Hello"
word.capitalize()
word.upper()
word.lower()
word.count("l")

'''
A function looks like this: function(something)
And a method looks like this: something.method()
All methods are functions, but not all functions are methods!
'''

# Packages
'''
For data science, the commonly used packages are:

Numpy: Working with arrays
Matplotlib: Data Visualisation
Scikit-learn: ML
Pandas: Data Manipulation
'''
# functions in modules
import math
math.pi
math.sqrt(25)
math.pow(2, 3)
dir(math)

import calendar
cal = calendar.month(2018, 1)
print(cal)

# import the functions.py file
import func
from func import calculate_triangle_area, calculate_rectangle_area
area = calculate_rectangle_area(5, 6) # call the function
print(area) # print the result

# modules < packages < libraries