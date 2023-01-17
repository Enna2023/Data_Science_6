# loops
# A loop is a sequence of instructions continuously repeated until a specific condition is reached

# for loop
# Looping through a List
a = ['banana', 'apple', 'mango']
for element in a:
    print(element)
    print(len(element))

b = [20, 10, 5]
total = 0
for e in b:
    total = total + e
print(total)

# range function
# range(start, stop, step)
range(1, 5)
c = list(range(1, 5))
print(c)


for i in range(1, 5):
    print(i)

total2 = 0
for i in range(1, 5):
    total2 += i
print(total2)

print(list(range(1,8)))

print(5 % 3)

total3 = 0
for i in range(1, 8):
    if i % 3 == 0:
        total3 += i
print(total3)
 
# compute the sum of the first 100 integers that are multiples of 3 or 5
list(range(1, 100))

# Initialize a variable to store the sum
sum = 0

# Iterate over the range of integers from 1 to 100
for i in range(1, 101):
  # If the current integer is a multiple of 3 or 5, add it to the sum
  if i % 3 == 0 or i % 5 == 0:
    sum += i

# Print the sum
print(sum)

'''
"For loop" is used for iterating over a sequence (that is either a list, a tuple, a dictionary, a set, or a string).

for variable in sequence:
   expression
'''
# Looping through a String
for letter in "Python":
    print(letter)

for i in "Python":
    print(i)

for i in "Python":
    print(i.capitalize())

# Enumerate function
# Enumerate function is used to loop over something and have an automatic counter.
# The enumerate() function takes a collection (e.g. a tuple) and returns it as an enumerate object.
# The enumerate object can then be used directly in for loops or be converted into a list of tuples using list() method.
# Syntax: enumerate(iterable, start=0)

for index, letter in enumerate("Python"):
    print(index, letter)

# while loop
# while loop is used to execute a set of statements as long as a condition is true.
# Syntax: while condition:
#             statement(s)

x = 1
while x < 5:
    print(x)
    x += 1


total = 0
i = 1
while i <= 5:
    total += i
    i += 1
print(total)

given_list = [5, 4, 4, 3, 1, -2, -3, -5]
total = 0
i = 0
while given_list[i] > 0:
    total += given_list[i]
    i += 1
print(total)

given_list = [5, 4, 4, 3, 1]
total = 0
i = 0
while i < len(given_list) and given_list[i] > 0:
    total += given_list[i]
    i += 1
print(total)


# use forloop instead of while loop add all the positive numbers in the list
given_list = [5, 4, 4, 3, 1, -2, -3, -5]
total = 0
for element in given_list:
    if element <= 0:
        break
    total += element
print(total)

# use while loop to add all the positive numbers in the list
total = 0
i = 0
while True:
    total += given_list[i]
    i += 1
    if given_list[i] <= 0:
        break
print(total)

# range function
   
for i in range(5):
    print(i, end=",")

for i in range(5, 10):
    print(i, end=" ")

'''
All three arguments are specified 
i.e., start = 0, stop = 10, step = 3. 
'''
for i in range(0, 10, 3):
    print(i, end=" ")

# break keyword
numbers = [5, 4, 4, 3, 1, -2, -3, -5]
for number in numbers:
    if number <= 0:
        print("Found a negative number")
        break
    print(number)

# List comprehension for the squares of all even numbers between 0 and 9
result = [x**2 for x in range(10) if x % 2 == 0]

print(result)

# loop

'''
for <temporary variable> in <list variable>:
  <action statement>
  <action statement>
'''
 
#each num in nums will be printed below
nums = [1,2,3,4,5]
for num in nums: 
  print(num, end=" ")

# continue keyword
big_number_list = [1, 2, -1, 4, -5, 5, 2, -9]

# Print only positive numbers:
for i in big_number_list:
  if i < 0:
    continue
  print(i, end=" ")

# loops with range

# Print the numbers 0, 1, 2:
for i in range(3):
  print(i)

# Print "WARNING" 3 times:
for i in range(3):
  print("WARNING")

# while loop

# This loop will only run 1 time
hungry = True
while hungry:
  print("Time to eat!")
  hungry = False

# This loop will run 5 times
i = 1
while i < 6:
  print(i, end=" ")
  i = i + 1

# nested loops
groups = [["Jobs", "Gates"], ["Newton", "Euclid"], ["Einstein", "Feynman"]]

# This outer loop will iterate over each list in the groups list
for group in groups:
  # This inner loop will go through each name in each list
  for name in group:
    print(name, end=" ")


# Print the First ten natural numbers using a while loop.
i = 1
while i <= 10:
    print(i, end=" ")
    i = i + 1

# Iterate over the following list and print the elements:
list1 = [12, 15, 32, 42, 55, 75, 122, 132, 150, 180, 200]
for i in list1:
    print(i, end=" ")
    
# Accept a number n from the user and print its multiplication table
n = int(input("Enter a number: "))
for i in range(2, 11):
    print(n, "x", i, "=", n*i)

'''
Use the enumerate function to print the elements of this list along with the indices:
grocery = ['bread', 'milk', 'butter']
'''
grocery = ['bread', 'milk', 'butter']
for index, item in enumerate(grocery):
    print(index, item)


# Take a number n from the user and find the sum of all numbers between 1 to n.
n = int(input("Enter a number: "))
sum = 0
for i in range(1, int(n)+1):
    sum += i
print('The sum of all numbers between 1 to',n,':',sum)

# Create a sequence of numbers from 3 to 5, and print each item in the sequence.
for i in range(3, 6):
    print(i, end=" ")

# Create a sequence of numbers from 3 to 19, 
# but increment by 2 instead of 1.
for i in range(3, 20, 2):
    print(i, end=" ")

# Print the letters of the string "Python" in the same line:
# Using a simple for loop
for i in "Python":
    print(i, end=" ")
# Using the range function
for i in range(6): # 6 is the length of the string "Python"
    print("Python"[i], end=" ")