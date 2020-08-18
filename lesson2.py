import tensorflow as tf

print(tf.version)

#this is a scalar value meaning it holds only one value:
scalar = tf.Variable("hello",tf.string);

#this is a vector meaning it holds a list of values:
vector = tf.Variable(["hello", "ali"], tf.string)

#this is a matrix meaning it holds a list containing one or more nested lists:
matrix = tf.Variable([["ali","kamel"],["keira","gratton"]],tf.string);

#the scalar value is of rank zero because it has zero lists
#the vector is of rank one because it has one list
#the matrix is of rank two because it has nested lists.

#we can determine the rank of a given tensorflow variable by running the following
print(tf.rank(scalar))#this will output a rank of 0
print(tf.rank(vector)) #this will ouput a rank of 1
print(tf.rank(matrix))#this will output a rank of 2

#this will show us the shape of the tensor:
print(matrix.shape) #this outputs (2,2) because there are 2 lists (corresponding to the first 2) and 2 values within them

tensor1 = tf.ones([1,2,3]) #this sets tensor1 to an array with 2 nested lists within which there are three value each
print(tensor1)
tensor2 = tf.reshape(tensor1, [2,3,1])
#this reshapes tensor2 so that there are 2 lists (one nested beneath the initial list) and 3 lists within the nested list
#within each of the 3 lists there is a one
print(tensor2)
tensor3 = tf.reshape(tensor1, [3,-1]) #when -1 is used to reshape a tensor this tells tensorflow to calculate what number
#would be here in this case it is 2 since there are 6 ones and 3*2 = 6
print(tensor3)

t = tf.zeros([5,5,5,5]) #this creates a matrix containing 625 values
print(t)
t =tf.reshape(t, [625]) #this flattens the matrix into a vector of 625 values
print(t)
t = tf.reshape(t, [125,-1]) #by using -1 in this situation tensorflow will calculate for us the appropriate value. (5)
print(t)




