import tensorflow as tf
#MNIST is a dataset of handwritten digits
from  tensorflow.examples.tutorials.mnist  import  input_data
MNIST  =  input_data.read_data_sets ( "~/data/mnist" ,  one_hot = True )
# setup learning parameters
batch_size = 100
training_epochs = 30
display_step = 1
learning_rate = 0.05

# create Tensor Flow graph inputs
# according to online docs MNIST had data image of shape 28px*28px=784px spaces
x = tf.placeholder(tf.float32, [None, 784])
# 10 classes for 0,1,..,9 character spaces
y = tf.placeholder(tf.float32, [None, 10])

# initialize model weights

#784 x 10 grid of 0's
W = tf.Variable(tf.zeros([784, 10]))
#10 grid of 0's
B = tf.Variable(tf.zeros([10]))

# Construct Logistic(Softmax) model
#I am using softmax regression, its a generalization of logistic regression that is heavily used on TF
model = tf.nn.softmax(tf.matmul(x, W) + B)

# Find minimum possible error using Cross-Entropy
#this step is similar to minizing RSS is linear regression modeling
cost_fcn = tf.reduce_mean(-tf.reduce_sum(y*tf.log(model), reduction_indices=1))
# Apply Gradient Decent
optimizing_fcn = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_fcn)

# Initialize all the variables
initialize = tf.global_variables_initializer()

# Start training
with tf.Session() as session:
    # Run:Initialize all the variables
    session.run(initialize)

    # loop to train the model
    for epoch in range(training_epochs):
        #initialize mean cost as a flt to avoid rounding issues
        mean_cost = 0.

        total_batches= int(MNIST.train.num_examples/batch_size)
        # Loop over each batch
        for i in range(total_batches):
            xs, ys = MNIST.train.next_batch(batch_size)
            # execute optimization and cost functions
            _, cost = session.run([optimizing_fcn, cost_fcn], feed_dict={x: xs,
                                                          y: ys})
            # Calculate mean loss for this batch
            mean_cost += cost / total_batches
        # Display logs per epoch step

    print("hey we're done")

    # Function to test if the 2 test and training agree
    correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Fucntion to evaluate accuracy
    accuracy_fcn = tf.reduce_mean(tf.cast(correct, tf.float32))
    #print out accuracy by evaluating model on test image set
    print(accuracy_fcn.eval({x: MNIST.test.images, y: MNIST.test.labels}))
