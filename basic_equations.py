from numpy import log, array, transpose, exp


# simple looking code to see if is easier to remember than math style equations

weights = array([1,2,3])
bias = 5
x = array([1,2,3])

def linear_regression(w, b, x):
    return transpose(w) * x + b

def sigmoid(z):
    return 1/(1+exp(-z))

def logistic_regression(w, b, x)
    return sigmoid(linear_regression(w, b, x))

def predict(x):
    # y_hat is probability y == 1 given x?
    # y_hat is predicted value of y given x?
    y_hat = probability(y, x)

def squared_error(predicted_output, correct_output):
    return 0.5 * pow((predicted_output-correct_output), 2)

def logistic_loss(predicted_output, correct_output):
    return -(correct_output * log(predicted_output) + (1-correct_output) * log(1 - predicted_output))

def cost(predicted_outputs, correct_outputs):
    total_loss = 0
    number_of_items = len(predicted_outputs)
    for index in range(0, number_of_items):
        total_loss += logistic_loss(predicted_outputs[index], correct_outputs[index])
    return total_loss / number_of_items

