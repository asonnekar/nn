import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)
weights_input_hidden = np.random.uniform(-1, 1, (2,4))
weights_hidden_output = np.random.uniform(-1, 1, (4,1))

bias_hidden = np.random.uniform(-1,1, (1,4))
bias_output = np.random.uniform(-1,1,(1,1))

learning_rate = 0.1

for epochs in range(10000):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input  = np.dot(hidden_layer_output, weights_hidden_output)
    output = sigmoid(output_layer_input)

    output_error = y-output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output += np.dot(hidden_layer_output.T, output_delta) * learning_rate
    weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate

    bias_hidden +=np.sum(hidden_delta, axis=0, keepdims = True) *learning_rate
    bias_output +=np.sum(output_delta, axis = 0, keepdims = True) * learning_rate
    if epochs %1000 == 0:
        print(f"Epoch {epochs}  Error = {np.mean(np.abs(output_error))}")

    print(f"Predicted Output: {output}")