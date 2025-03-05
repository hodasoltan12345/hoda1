def sigmoid(x):
    return 1 / (1 + (2.718281828459045 ** -x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = [[0.05, 0.10]]

w = [
    [0.15, 0.20],
    [0.25, 0.30],
    [0.40, 0.45],
    [0.50, 0.55]
]

b_hidden = 0.35
b_output = 0.60

targets = [0.01, 0.99]

h1_input = sum(inputs[0][i] * w[0][i] for i in range(2)) + b_hidden
h1_output = sigmoid(h1_input)
h2_input = sum(inputs[0][i] * w[1][i] for i in range(2)) + b_hidden
h2_output = sigmoid(h2_input)

o1_input = h1_output * w[2][0] + h2_output * w[2][1] + b_output
o1_output = sigmoid(o1_input)
o2_input = h1_output * w[3][0] + h2_output * w[3][1] + b_output
o2_output = sigmoid(o2_input)

learning_rate = 0.5

delta_o1 = (o1_output - targets[0]) * sigmoid_derivative(o1_output)
delta_o2 = (o2_output - targets[1]) * sigmoid_derivative(o2_output)

w[2][0] -= learning_rate * delta_o1 * h1_output
w[2][1] -= learning_rate * delta_o1 * h2_output
w[3][0] -= learning_rate * delta_o2 * h1_output
w[3][1] -= learning_rate * delta_o2 * h2_output

b_output -= learning_rate * (delta_o1 + delta_o2)

delta_h1 = (w[2][0] * delta_o1 + w[3][0] * delta_o2) * sigmoid_derivative(h1_output)
delta_h2 = (w[2][1] * delta_o1 + w[3][1] * delta_o2) * sigmoid_derivative(h2_output)

w[0][0] -= learning_rate * delta_h1 * inputs[0][0]
w[0][1] -= learning_rate * delta_h1 * inputs[0][1]
w[1][0] -= learning_rate * delta_h2 * inputs[0][0]
w[1][1] -= learning_rate * delta_h2 * inputs[0][1]

b_hidden -= learning_rate * (delta_h1 + delta_h2)

h1_input = sum(inputs[0][i] * w[0][i] for i in range(2)) + b_hidden
h1_output = sigmoid(h1_input)
h2_input = sum(inputs[0][i] * w[1][i] for i in range(2)) + b_hidden
h2_output = sigmoid(h2_input)

o1_input = h1_output * w[2][0] + h2_output * w[2][1] + b_output
o1_output = sigmoid(o1_input)
o2_input = h1_output * w[3][0] + h2_output * w[3][1] + b_output
o2_output = sigmoid(o2_input)

print(round(o1_output, 2), round(o2_output, 2))
