

'''
A shortcut for adding comments is by using CTRL + /
Simply select all the lines you want to comment out, then press Ctrl + /, and all the lines will get commented out!
''' 
# strings are surrounded by either single quotation marks, or double quotation marks.
a = "Hello, World!"
print(a[-4])
print(a[-5:-2])

x = "Python is "
y = "awesome"
z =  x + y
print(z)

print(len(a))

# lists are surrounded by square brackets
'''
Lists are mutable. This means that you can change the values of the list after it has been created.
Some essential features of Python lists are:
Collection of values
Can be of any data type
Can be a combination of different types
'''
items = ["apple", "banana", "cherry"]
print(items)
print(items[1])
print(items[-1])
items[1] = "blackcurrant"
print(items)
items[0:2]
items.append("orange")
print(items)
items.insert(1, "lemon")
print(items)
items.remove("lemon")
print(items)

food= ["apple", "banana", "cherry", "orange", "kiwi", "melon", "mango"]
bathroom = ["toothpaste", "toothbrush", "soap", "shampoo", "conditioner"]
# shopping_list = [food, bathroom]
shopping_list = food + bathroom
print(shopping_list)
len(shopping_list)

'apple' in shopping_list
'pear' in shopping_list

# lists methods
# len() returns the length of the list
# .append() adds an item to the end of the list
# .insert() adds an item at the specified index
# .remove() removes the specified item
# .pop() removes the item at the specified index
# .clear() removes all the items from the list
# .index() returns the index of the specified item
# .count() returns the number of items with the specified value
# .sort() sorts the list
# .reverse() reverses the order of the list
# .copy() copies the list
# .extend() adds the specified list elements to the end of the current list
# .join() returns a string concatenated with the elements of an iterable
# .split() returns a list where the string has been split at each match

# tuples are surrounded by parentheses
'''
The only difference between Tuple & List is that Tuple is immutable; once created, it cannot be changed.
Example: A = ('Brush', 'Leuven', 48851964400, 3.14)
'''
x = ("apple", "banana", "cherry")
print(x)
x[0]
x.count("apple")
len(x)
y = (1,'a',2,'b')
z = x + y
print(z)
a = ('Hi',) * 5
print(a)
max(x)
del z
