# Conditional Statement and Dictionaries

# Conditional Statement: if-elif-else

'''if  condition1:
    # do something
'''
if 5 > 2:
    print("Five is greater than two!")

if 3 < 2:
    print("Three is less than two!")
else:
    print("The condition is not true!")

3!=2

age = 18
if age < 18:
    print("You are not old enough to vote!")   
elif age == 18:
    print(" You are old enough to vote!")
else:
    print("You are old enough to vote!")

if (5 < 18) or (101 > 100):
    print("Hi")

'''
the program evaluates the test expression and will execute statement(s) only if the test expression is True.
If the test expression is False, the statement(s) is not executed.
'''
if (5 > 18) and (101 > 100):
    print("Hi")
else:
    print("Hello")
# Dictionaries are a way to store data in key-value pairs
# Dictionaries are mutable

'''
if test expression:
   Body of if 
else:
   Body of else
'''
z = 5
if z%2 == 0:
    print("z is even")
else:
    print("z is odd")

# if-elif-else

'''
if test expression:
   Body of if
elif test expression:
   Body of elif
else:
   Body of else

The elif is short for else if. It allows us to check for multiple expressions.
If all the conditions are False, the body of else is executed.
Only one block among the several if-elif-else blocks is executed according to the condition.
The if block can have only one else block. But it can have multiple elif blocks.
'''
z=5
if z%2 == 0:
    print("z is even")
elif z%3 == 0:
    print("z is divisible by 3")
else:
    print("z is odd")

# dictionary

'''
A dictionary is a collection which is unordered, changeable and indexed, unordered collection of key-value pairs. In Python dictionaries are written with curly brackets, and they have keys and values.

my_dict = {
key1:value1,
key2:value2,
}

The keys in a dictionary must always be unique and immutable. This is the reason dictionary keys can be String but not List.
On the other hand, Values in a dictionary can be of any datatype and can be duplicated
'''

d = {}
d['George'] = 24
d['Tom'] = 32
d['Jenny'] = 16
print(d)
d["Jenny"] = 20
print(d)

# Iterating through a dictionary
for key in d:
    print(key, d[key])

world = {"afghanistan":30.55, "albania":2.77, "algeria":39.21}
for key, value in world.items():
    print(key + " - " + str(value))

# Take values of length and breadth of a rectangle from the user and check if it is a square.

length = float(input("Enter the length: "))
breadth = float(input("Enter the breadth: "))
if length == breadth:
    print("It is a square")
else:
    print("It is a rectangle")

# Take two int values from the user and print greatest among them.
value1 = float(input("Enter the first value: "))
value2 = float(input("Enter the second value: "))
if value1 > value2:
    print("The first value is greater than the second value")
elif value1 < value2:
    print("The second value is greater than the first value")
else:
    print("Both values are equal")

# Write a program to read a candidate's age and determine whether they are eligible to cast their vote.
candidate_age = int(input("Enter the candidate's age: "))
if candidate_age >= 18:
    print("The candidate is eligible to cast their vote")
else:
    print("The candidate is not eligible to cast their vote")

# Write a Python program to add a key to a dictionary.
'''Sample Dictionary : {0: 10, 1: 20}
Expected Result : {0: 10, 1: 20, 2: 30}'''
d = {0:10, 1:20}
d[2] = 30
print(d)

'''
Below are two lists; convert them into a dictionary. keys = ['Ten', 'Twenty', 'Thirty']
values = [10, 20, 30]
Expected output:
{'Ten': 10, 'Twenty': 20, 'Thirty': 30}
'''
keys = ['Ten', 'Twenty', 'Thirty']
values = [10, 20, 30]
d = dict(zip(keys, values))
print(d)

# Access the value of key 'history'
sampleDict = {
"class":{
"student":{
"name":"Mike",
"marks":{
"physics":70,
"history":80
}
}
}
}
history_marks = sampleDict.get("class").get("student").get("marks").get("history")
history_marks = sampleDict["class"]["student"]["marks"]["history"]

print(history_marks)


# Given the following dictionary:

inventory = {
'gold' : 500,
'pouch' : ['flint', 'twine', 'gemstone'],
'backpack' : ['xylophone','dagger', 'bedroll','bread loaf']
}

# Add a key to inventory called 'pocket'.
inventory["pocket"] = []

# Set the value of 'pocket' to be a list consisting of the strings 'seashell', 'strange berry', and 'lint'.
inventory["pocket"] = ["seashell", "strange berry", "lint"]

# Sort the items in the list stored under the 'backpack' key.
inventory["backpack"].sort()
print(inventory["backpack"])

# Remove 'dagger' from the list of items stored under the 'backpack' key.
inventory["backpack"].remove("dagger")
print(inventory["backpack"])

# Add 50 to the number stored under the 'gold' key.
inventory["gold"] += 50
print(inventory["gold"])