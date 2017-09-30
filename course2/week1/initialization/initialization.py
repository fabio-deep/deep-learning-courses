import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import sklearn.datasets
from init_utils import *
matplotlib.use('TkAgg')

#plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()

'''
A well chosen initialization can:
   -Speed up the convergence of gradient descent
   -Increase the odds of gradient descent converging to a lower training (and generalization) error

You will use a 3-layer neural network (already implemented for you). Here are the initialization methods you will experiment with:
   -Zeros initialization -- setting initialization = "zeros" in the input argument.
   -Random initialization -- setting initialization = "random" in the input argument. This initializes the weights to large random values.
   -He initialization -- setting initialization = "he" in the input argument. This initializes the weights to random values scaled according to a paper by He et al., 2015.'''

def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent 
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    Returns:
    parameters -- parameters learnt by the model
    """
        
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]
    
    # Initialize parameters dictionary.
    if initialization == "zeros":
    	print("\nZero initialization:")
    	parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
    	print("\nRandom initialization:")
    	parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
    	print("\nHe initialization:")
    	parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        
        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

'''
There are two types of parameters to initialize in a neural network:
   -the weight matrices  (W[1],W[2],W[3],...,W[L−1],W[L])
   -the bias vectors     (b[1],b[2],b[3],...,b[L−1],b[L])

Exercise: Implement the following function to initialize all parameters to zeros. 
You'll see later that this does not work well since it fails to "break symmetry", but lets try it anyway and see what happens. 
Use np.zeros((..,..)) with the correct shapes.'''

def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims)            # number of layers in the network
    
    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###
    return parameters

parameters = initialize_parameters_zeros([3,2,1])
print("\nZero initialization of params for network with 3 input, 2 hidden, 1 output:")
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]) + "\n")

parameters = model(train_X, train_Y, initialization = "zeros")
print ("\nOn the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("\nOn the test set:")
predictions_test = predict(test_X, test_Y, parameters)

print("\nBad predictions:")
print ("predictions_train = " + str(predictions_train))
print ("predictions_test = " + str(predictions_test))


plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
plt.show()

'''
The model is predicting 0 for every example.
In general, initializing all the weights to zero results in the network failing to break symmetry. 
This means that every neuron in each layer will learn the same thing, and you might as well be training a neural network with  n[l]=1 for every layer, 
and the network is no more powerful than a linear classifier such as logistic regression.'''

'''
What you should remember:
   -The weights  W[l]  should be initialized randomly to break symmetry.
   -It is however okay to initialize the biases  b[l] to zeros. 
   -Symmetry is still broken so long as  W[l] is initialized randomly.'''

'''
Exercise: Implement the following function to initialize your weights to large random values (scaled by *10) and your biases to zeros. 
Use np.random.randn(..,..) * 10 for weights and np.zeros((.., ..)) for biases. 
We are using a fixed np.random.seed(..) to make sure your "random" weights match ours, 
so don't worry if running several times your code gives you always the same initial values for the parameters.'''

def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers
    
    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        ### END CODE HERE ###

    return parameters

parameters = initialize_parameters_random([3, 2, 1])
print("\nLarge value random initialization:")
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


parameters = model(train_X, train_Y, initialization = "random")
print ("\nOn the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("\nOn the test set:")
predictions_test = predict(test_X, test_Y, parameters)

print("\nPredictions with large value random initialization:")
print (predictions_train)
print (predictions_test)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
plt.show()

'''
In summary:
Initializing weights to very large random values does not work well.
Hopefully intializing with small random values does better. 
Poor initialization can lead to vanishing/exploding gradients, which also slows down the optimization algorithm.
The important question is: how small should be these random values be? Lets find out in the next part!

Exercise: Implement the following function to initialize your parameters with He initialization.

This is similar except Xavier initialization uses a scaling factor for the weights  W[l]  of sqrt(1./layers_dims[l-1]) 
where He initialization would use sqrt(2./layers_dims[l-1]).)

This function is similar to the previous initialize_parameters_random(...). 
The only difference is that instead of multiplying np.random.randn(..,..) by 10, 
you will multiply it by  √2/dimension of the previous layer, 
which is what He initialization recommends for layers with a ReLU activation.'''

def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        ### END CODE HERE ###
        
    return parameters

parameters = initialize_parameters_he([2, 4, 1])
print("\nHe initialization params:")
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = model(train_X, train_Y, initialization = "he")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
plt.show()

'''
Observations:
The model with He initialization separates the blue and the red dots very well in a small number of iterations.

For the same number of iterations and same hyperparameters the comparison is:

                                      Model|	 Train accuracy|	          Problem/Comment|
===========================================|===================|=============================|
       3-layer NN with zeros initialization|	            50%|	  fails to break symmetry|
3-layer NN with large random initialization|	            83%|	        too large weights|
          3-layer NN with He initialization|	            99%|	       recommended method|


What you should remember:
   -Different initializations lead to different results
   -Random initialization is used to break symmetry and make sure different hidden units can learn different things
   -Don't intialize to values that are too large
   -He initialization works well for networks with ReLU activations. '''