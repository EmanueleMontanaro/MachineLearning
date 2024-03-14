import numpy

print('Creating arrays and using main properties')
x = numpy.array([1,2,3])

y = numpy.array([[1,2,3], [4,5,6]])

print(x.size,y.size)
print(y.ndim)
print(x.shape,y.shape)

print('\nCreating empty array')
x = numpy.zeros((2, 3), dtype=numpy.float32)
print(x)

print('\nCreating ones array')
x = numpy.ones(5)
print(x)

print('\nCreating increasing array')
x = numpy.arange(4)
print(x)
print('\nIt allows a specific step size')
x = numpy.arange(0, 6, 2)
print(x)
print('\nCreating array of evenly spaced values')
x = numpy.linspace(0, 5, 4)
print(x)

print('\nCreating identity matrix')
x = numpy.eye(3)
print(x)

print('\nBasic operations between ararys, require matching shapes')
x = numpy.array([[1,2,3],[4,5,6]])
y = numpy.array([[2,2,2],[3,3,3]])
print('x + y =', x + y)
print('x * y = ',x*y)
print('\nCan be used as x+=y or x*=y to modify x array in-loco')

print('\nMatrix product requires matching columns with rows')
x = numpy.array([[1,2],[3,4],[5,6]])
y = numpy.array([[1,2,3],[4,5,6]])
print(numpy.dot(x, y))

print('numpy arrays can be reshaped')
x = numpy.array([[1,2],[3,4],[5,6]])
y = x.reshape((2,3))
print('The order of data is preserved')
print('Reshaping to one-dimensional array: (NOT A SINGLE ROW VECTOR)')
print(x.ravel())

print('\nTransposing:')
x = numpy.arange(3).reshape(3,1)
print(x.T)

print('\nOperations to reduce, they operate as if the array was one-dimensional')
x = numpy.array([[1,2,3],[4,5,6]])
print(x.sum())
print(x.max())
print('It is possible to specify a single axis on which to perform:')
print(x.sum(axis=0))

print('\nCreating a new array with element-wise functions:')
x = numpy.array([[1,2,3],[4,5,6]])
print(numpy.exp(x))
print(numpy.log(x))

print('\nArrays can be sliced:')
x = numpy.arange(5)
print(x[1:3])
print(x[::2])
print(x[3])
print('For multidimensional arrays we need to specify the axis')
x = numpy.arange(15).reshape(3,5)
print(x[1, 0:3])
print('\nWhen slicing we create a view, an array\nthat shares its data with a different one.\nModification to a view modifies the original array')
x = numpy.arange(5)
print(x)
y=x[0:3]
y[:] = 3
print(x)

