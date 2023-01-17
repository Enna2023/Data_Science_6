import numpy as np
from timeit import Timer

# 2.Creating Numpy array from Python list

# one-dimensional array
heights = [165, 170, 171, 180, 189, 178]
print(type(heights))

heights_np = np.array(heights)
print(type(heights_np))

# two-dimensional array
weights = np.array([[50, 45, 56, 78],[78, 89, 59, 90],[89, 78, 69, 70],[67, 69, 89, 70],[90,89, 80, 84],[89, 59, 90, 78]])
print(weights)

# 3. Exploring some of the key attributes of ndarray objects

'''
ndim: number of dimensions of the array
shape: shape of the array in the format (number_rows, number_columns)
size: total number of elements
dtypes: type of data stored in the array
strides: number of bytes that must be moved to store each row and column in memory, in the format (number_bytes_files, number_bytes_columns)
'''

print("dimension:", weights.ndim)
print("shape:", weights.shape)
print("size:", weights.size)
print("dtype:", weights.dtype)
print("strides:", weights.strides)

# Convert the two-dimensional ndarray weights into a three-dimensional object without changing its shape.
weights_3d = weights.reshape(8, 3, 1) 
print(weights_3d)

# Convert weights into a three-dimensional object
weights_3d = np.expand_dims(weights, axis=2)
# Print the shape of the resulting array
print(weights_3d.shape)


# 4. Creating arrays with specific values
# np.zeros(shape, dtype=float, order='C')
x = np.zeros(shape=(3,5), dtype ="int32")
print(x)

# 4.2. np.arange()
# np.arange(start, stop, step, dtype=None)
x = np.arange(0, 10, 2)
print(x)
y = np.arange(10)
print(y)

# 4.3. np.linspace() 
# Return evenly spaced numbers over a specified interval.
# np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

x = np.linspace(0, 10, 5)
print(x)

# 4.4. np.full()
# Return a new array of given shape and type, filled with fill_value.
# np.full(shape, fill_value, dtype=None, order='C')
x = np.full((3, 5), 5)
print(x)


# 4.5. Creating arrays with random values
# Return random floats in the half-open interval [0.0, 1.0).
# np.random.random()
x = np.random.random((3, 5))
print(x)

#5 resize, reshape, ravel, flatten, squeeze, expand_dims

#5.1. reshape
weights = np.array([[50, 45, 56, 78],[78, 89, 59, 90],[89, 78, 69, 70],[67, 69, 89, 70],[90,89, 80, 84],[89, 59, 90, 78]])
print(weights)
weights_3d = weights.reshape(2, 3, 4)
print(weights_3d)

weights_3d = weights.reshape(8, 3)
print(weights_3d)

#5.2. resize
# Return a new array with the specified shape.
# np.resize(a, new_shape)
x = np.arange(10)
print(x)
x = np.resize(x, (2, 6))
print(x)

# 5.3. falten
# Return a copy of the array collapsed into one dimension.
weights_faltten = weights.flatten()
print(weights_faltten)

#5.3. ravel
# Return a contiguous flattened array.
weights_ravel = weights.ravel()
print(weights_ravel)

'''
Create an array of 51 elements starting at 100 and ending at 500, using the two functions np.linspace() and np.arange(). Arrays must have the same content, with the names array_lin and array_ara, respectively. Verify that the arrays have the same content with the np.array_equal() function.
'''
array_lin = np.linspace(100, 500, 51)
array_arange = np.arange(100, 501, 10)
print(array_lin)
print(array_arange)
print(np.array_equal(array_lin, array_arange))

# 6. array indexing
# positive indexing
weights = weights.reshape((2, 6, 2))
print(weights)
weights_or = weights.reshape((6,4))
print(weights_or)
print(heights_np)
print("Accessing single element in 1D array:", heights_np[2])
print("Accessing single element in 2D array:", weights_or[1][3])

print(heights_np[:4]) # The default start value is 0
print(heights_np[2:]) # The default end value is the last value of the array
print(heights_np[1:4]) # The ending value is exlusive

print(weights_or[:2, :2])
print(weights_or[:2, ::2])
print(weights_or[:2, 1::2])
print(weights_or[:3, 3:4:])    # 3:4: means 3 to 4-1

# negative indexing
heights_np[:-4] # Equivalent to heights_np[:2]
heights_np[-4:] # Equivalent to heights_np[2:]
heights_np[-4:-1] # Equivalent to heights_np[2:5]
weights[:2, -3::] # Equivalent to weights[:3, 3::]
weights[:3, :-3, :-1] # Equivalent to weights[:3, :3, :1]


# Consider the weights_or array:

# Select all the values that are in the even positions in the rows and in the odd positions in the columns. Create a new array named weights_custom1 with these values.
weights_custom1 = weights_or[:, ::2]
print(weights_custom1)
# Express the weights_custom1 array flattened with an in-memory copy. Call the new array weights_custom2.
weights_custom2 = weights_custom1.flatten()
print(weights_custom2)
# Select items in positions 2 to 4 inclusive with negative indexing. Name the output array as weights_custom3.
weights_custom3 = weights_custom2[-4:-1]
print(weights_custom3)




# 7.Manipulating Numpy arrays
# 7.1. Arithmetic operations 
# mean, median, std, var, min, max, sum, cumsum, cumprod, diff, sort, argsort, argmin, argmax, round, clip, prod, mean, median, std, var, min, max, sum, cumsum, cumprod, diff, sort, argsort, argmin, argmax, round, clip, prod


heights_2 = np.array([165, 175, 180, 189, 187, 186])
print('heights_np:', heights_np) 
print('heights_2: ', heights_2)

heights_add = heights_np + heights_2
print('heights_add: ', heights_add)

# element-wise elements of one-dimensional arrays
added = np.add(heights_2, heights_np)
print('added: ', added)

heights_sub = np.subtract(heights_2, heights_np)   # Subtract heights_2 from heights_np
print('heights_sub: ', heights_sub)

heights_mul = np.multiply(heights_2, heights_np)   # Multiply heights_2 by heights_np
print('heights_mul: ', heights_mul)

heights_div = np.divide(heights_2, heights_np)   # Divide heights_2 by heights_np
print('heights_div: ', heights_div)


# Calculate the product element-wise of the multiplicative inverses ( 1/x ) between the arrays heights_np and heights_2, using numpy functions. 
# Name the output array as heights_inv.
heights_inv = np.divide(1, heights_np) * np.divide(1, heights_2)
print('heights_inv: ', heights_inv)

sin = np.sin(heights_np)
print('sin: ', sin)

# 7.2. logical operations

x = np.arange(5)
print(x)
np.logical_or(x < 1, x > 3)
np.logical_and(x ==3 , x == 4)

x = np.array([True, True, False, False])
y = np.array([True, False, True, False])
np.logical_or(x, y)
np.logical_and(x, y)
np.logical_not(x)

# 7.3 Comparison - Relational operators
x = np.array([1, 8, 3, 7, 3, 21])
y = np.array([4, 8, 1, 7, 6, 9])
np.equal(x,y)
np.not_equal(x,y)
np.less_equal(x,y) # x <= y
np.greater_equal(x,y) # x >= y
np.array_equal(x,y) # Comparing the entire content of both arrays
x = np.array([1, 8, 3, 7, 3, 21])
y = np.array(list((1, 8, 3, 7, 3, 21)))
np.array_equal(x,y) # Comparing the entire content of both arrays


# 8. Broadcasting
heights_np = heights_np.reshape((6,1))
print(heights_np)
print(weights)

# We are going to add the elements of both arrays:
broad_np = heights_np + weights
print(broad_np)


x = np.ones((3,4))
y = np.random.random((5,1,4))
print(y)
z = x + y
print(z)

import numpy as np
# Propose an array y such that the operation  x+y  results in the array z.
x = [[14, 15, 18],
    [62, 90, 98],
    [71, 73, 90],
    [40, 24, 17],
    [11, 81, 14],
    [26, 81, 31]]

z = [[24,  40,  58],
    [72, 115, 138],
    [81,  98, 130],
    [50,  49,  57],
    [21, 106,  54],
    [36, 106,  71]]

y = np.subtract(z, x)
print(y)

# 9.Matrix multiplication
A = np.array([[1,1,8],[0,1,9],[9,0,8]])
print("Matrix A:\n", A, '\n')

B = np.array([[2,0,0],[3,4,9],[7,8,9]])
print('MATRIX B:\n', B, '\n')

print("Element wise multiplication:\n", A*B, '\n')

# The dot product of matrices can be executed with the @ operator or with the numpy np.dot() function:
print("Matrix product:\n", A@B, '\n') 
# matrix A = (2 ,3) , matrix B= (3,4), output matrix =( 2,4)
print("Dot product:\n", A.dot(B), '\n')

# 10. Arrays with random numbers

'''
np.random.random(): returns random floats in the half-open interval [0.0, 1.0)
np.random.randint(low, high): returns random integers from low (inclusive) to high (exclusive).
np.random.normal(): returns random samples from a normal (Gaussian) distribution.
np.random.choice(): returns a random sample from a given 1-D array.'''

np.random.random((4,3))

np.random.randint(10, 20, size=(2, 4))

np.random.normal(size=10)

a = np.ones((4, 3), dtype=int)
b = np.random.random((4, 3))
b += a
print(b)

from numpy.random import seed
from numpy.random import rand

# Seed random number generator
seed(42)

# Generate random numbers between 0-1
values = rand(10)
print(values)
rand_1 = np.random.randint(10) # Generate a random integer between 0 and 10
print(rand_1)
gauss = np.random.normal(100) # Generate a random number from a normal distribution with mean 100
print(gauss)

# 11. Concatenate, and stack Numpy arrays
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
print(np.concatenate((a, b), axis=0))

my_array = np.array([1,2,34,5])
x = np.array([1,4,5,6])
print('x: \t  ', x)
print('my_array: ', my_array)

print('Append:\n',np.append(my_array,x))
y = np.append(my_array, x)

# Concatentate `my_array` and `x`
print('\nConcatenate:\n',np.concatenate((my_array,x)))

# Stack arrays vertically (row-wise)
print("Stack row wise:")
print(np.vstack((my_array, x)))

# Stack arrays horizontally
print("Stack horizantally:")
print(np.hstack((my_array,x)))

print("\nAnother way:")
print(np.r_[my_array,x])

# Stack arrays column-wise
print("Stack column wise:")
print(np.column_stack(( my_array,x)))

print("\nColumn wise repeat:")
print(np.c_[ my_array,x])

# 12. Visualize Numpy arrays

import matplotlib.pyplot as plt

#  specify an initial state for the Mersenne Twister number generator, a pseudo-random number generator:

rng = np.random.RandomState(10)  

# Now we generate random values of two normal distributions with different mean and standard deviation, a distribution of mean 0 and another of mean 5, stacking them in a single array horizontally:

a = np.hstack((rng.normal(size=1000),rng.normal(loc=5, scale=2, size=1000)))
print(a)

plt.hist(a, bins='auto')
plt.title("Histogram")
plt.show()

# As can be seen, this graph denotes the distribution of the two normal distributions with a mean of 0 and 5.

# As an additional example, we are creating a meshgrid np.meshgrid() with values generated from an array of numpy with initial value of 5, final value of -5 (exclusive) and step of 0.01. We have calculated the value of z which corresponds to the general equation of a circle, so that we can generate the graph shown below:

# Create an array
points = np.arange(-5, 5, 0.01)

# Make a meshgrid
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs ** 2 + ys ** 2)

# Display the image on the axes

# Display the image on the axes
plt.imshow(z, cmap=plt.cm.Reds)

# Draw a color bar
plt.colorbar()

# Show the plot
plt.show()

# 13. Save the numpy ndarray object into a npy file
# one of the most important parts of the entire analysis process, the storage of the results. We can do this with the np.savetxt() function:

import numpy as np
x = np.arange(0.0,5.0,1.0)
np.savetxt('test.txt', x, delimiter=',')

