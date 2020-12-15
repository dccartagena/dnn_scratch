import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import h5py

# Class layer
class layer:
    def __init__(self, size_input, size_output, activation):
        # Class constructor
        self.size_input = size_input
        self.size_output = size_output

        self.input      = np.zeros([size_input, 1])
        self.output     = np.zeros([size_output, 1])
        self.preact     = np.zeros([size_output, 1])

        self.weight     = np.random.randn(size_output, size_input)
        self.bias       = 1e-3 * np.random.randn(size_output, 1)

        self.d_dweight  = np.zeros([size_output, size_input])
        self.d_dbias    = np.zeros([size_output, 1])

        self.loss       = f_x_entropy

        self.act_dict   = { "relu"      : f_relu, 
                            "tanh"      : f_tahn,
                            "softmax"   : f_softmax}
        self.activation = self.act_dict[activation]
        pass

    def set_parameter(self, weight, bias):
        # set parameters
        self.weight = weight
        self.bias   = bias
        pass

    def set_input(self, input):
        # Set input
        self.input  = input
        pass

    def set_output(self):
        # Compute layer output
        self.preact = np.matmul(self.weight, self.input) + self.bias
        self.output = self.activation(self.preact)

    def get_gradient_act(self):
        # Compute gradient
        return self.activation(self.preact, True)

    def set_gradient(self, new_epoch = True, d_dweight = [], d_dbias = [], batch_size = 1):
        # Compute gradients for parameters
        if (new_epoch == True):
            self.d_dweight  = np.zeros([self.size_output, self.size_input])
            self.d_dbias    = np.zeros([self.size_output, 1])
        else:
            self.d_dweight  += np.divide(d_dweight, batch_size)
            self.d_dbias    += np.divide(d_dbias, batch_size)
        pass

    def f_sgd(self, learn_rate):
        # Optimizer: Stochastic gradient descent
        self.weight -= learn_rate * self.d_dweight
        self.bias   -= learn_rate * self.d_dbias
        pass

# Activation functions
def f_tahn(x, derivative = False):
    # tanh
    if (derivative):
        return 1 - np.power(np.tanh(x), 2)
    return np.tanh(x)

def f_relu(x, derivarive = False):
    # ReLU
    if (derivarive):
        return np.multiply(1, np.array(x > 0))
    return np.multiply(x, np.array(x > 0))

def f_softmax(x, derivative = False):
    # Softmax
    shiftx = x - np.max(x)
    softmax = np.exp(shiftx) / sum(np.exp(shiftx))
    if (derivative):
        return  1
    return softmax

# Loss function
def f_x_entropy(prediction, target, derivative = False):
    # Cross-entropy
    target_size = target.shape[0]
    if (derivative):
        return  prediction - target 
    return - (1 / target_size) * np.sum(target * np.log(np.maximum(prediction, 1e-12)))

# Accuracy evaluation
def f_accuracy(y_pred, y_true):
    # Accuracy
    y_logic = 1 * (y_pred == max(y_pred))
    true_samples = 1 * (y_logic == y_true).all(axis = 0)
    return true_samples

# Feedfordward method
def f_feedfordward(model, data, target):
    # Set data
    model[0].set_input(data)

    # Update outputs
    for i in range(len(model)):
        model[i].set_output()

        if (i == (len(model) - 1)):
            prediction = model[-1].output
            loss    = f_x_entropy(prediction, target)
            acc     = f_accuracy(prediction, target)
            return model, prediction, loss, acc

        model[i + 1].set_input(model[i].output)

# Compute gradients  
def get_gradient(model, data, target, batch_size):

    prediction = model[-1].output
    back_gradient = f_x_entropy(prediction, target, derivative = True)
    back_weight = np.eye(model[-1].output.shape[0])

    for i in reversed(range(len(model))):
        act_gradient = model[i].get_gradient_act()
        back_gradient =  np.multiply(act_gradient, np.matmul(back_weight, back_gradient))

        back_weight = model[i].weight.T

        weight_gradient = np.matmul(back_gradient, model[i].input.T)
        bias_gradient = back_gradient

        model[i].set_gradient(False, weight_gradient, bias_gradient, batch_size)

    return model

def f_plot(epochs, hist, type, parameter, learn_rate, activation):
    div_epoch = np.array([20, 40])
    learn_change = hist[div_epoch]

    title = '{} {} with lr: {}, activation: {}'.format(type, parameter, learn_rate, activation)
    
    fig = plt.figure()
    plt.plot(epochs, hist)
    plt.scatter(div_epoch, learn_change)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(parameter)
    plt.grid(True)
    # plt.show()

    filename_figure = 'figures/fig_{}_{}_{}_{}.png'.format(activation, learn_rate, parameter, type)
    fig.savefig(filename_figure)

    plt.close(fig)
    pass

# DNN
def d_nn(learn_rate = 1e-1, activation = 'relu', max_epoch = 10, batch_size = 280, verbose = True):

    # Learning rate decay
    div_epoch = np.array([20, 40])

    # Dataset
    file_name   = 'mnist_testdata.hdf5'
    dataset     = h5py.File(file_name, 'r')

    xdata = dataset.get('xdata').value
    ydata = dataset.get('ydata').value

    # Training, validation and test proportions
    training_size   = int(0.7 * xdata.shape[0])
    validation_size = int(0.2 * xdata.shape[0])
    test_size       = int(0.1 * xdata.shape[0])
    
    updates = int(training_size / batch_size)

    # Spliting data
    train_x, val_x, test_x = np.split(xdata, [training_size, training_size + validation_size], axis = 0)
    train_y, val_y, test_y = np.split(ydata, [training_size, training_size + validation_size], axis = 0)

    feature_size    = xdata.shape[1]
    target_size     = ydata.shape[1]

    # Training accuracy and loss history
    acc_train_hist  = np.zeros([max_epoch])
    loss_train_hist = np.zeros([max_epoch])

    # Validation accuracy and loss history
    acc_val_hist  = np.zeros([max_epoch])
    loss_val_hist = np.zeros([max_epoch])

    # Structure of the NN
    structure = (feature_size, 128, target_size)
    
    # Initialize NN
    layer_nn = []
    for i in range(len(structure) - 1):
        if i == len(structure) - 2:
            activation = 'softmax'
        layer_nn.append(layer(structure[i], structure[i + 1], activation))

    # Training
    for i in range(max_epoch):

        # Learning decay
        if np.any(i == div_epoch):
            learn_rate = learn_rate / 2
        
        for k in range(updates):
            for j in range(batch_size):    
                data_cache = np.reshape(train_x[k + j, :], [feature_size, 1])
                target_cache = np.reshape(train_y[k + j, :], [target_size, 1])

                # Feedfordward
                layer_nn, _, loss_train_cache, acc_train_cache = f_feedfordward(layer_nn, data_cache, target_cache)

                # Get gradients
                layer_nn = get_gradient(layer_nn, data_cache, target_cache, batch_size)

                acc_train_hist[i]  += np.divide(acc_train_cache, training_size)
                loss_train_hist[i] += np.divide(loss_train_cache, training_size)

            # Update NN
            for l in range(len(structure) - 1):
                # Optimizer
                layer_nn[l].f_sgd(learn_rate)

                # Initialize gradient for new epoch
                layer_nn[l].set_gradient()

        # Compute loss and accuracy for validation
        for j in range(validation_size):

            data_cache = np.reshape(val_x[j, :], [feature_size, 1])
            target_cache = np.reshape(val_y[j, :], [target_size, 1])

            _, _, loss_train_cache, acc_train_cache = f_feedfordward(layer_nn, data_cache, target_cache)

            acc_val_hist[i] += np.divide(acc_train_cache, validation_size)
            loss_val_hist[i] += np.divide(loss_train_cache, validation_size)

        # Print results
        if (verbose):
            print("epoch: {} -- loss train: {} -- acc train: {} -- loss val: {} -- acc val: {} \n".format(i, loss_train_hist[i], acc_train_hist[i], loss_val_hist[i], acc_val_hist[i]))

    acc_test = 0
    for j in range(test_size):
        data_cache = np.reshape(test_x[j, :], [feature_size, 1])
        target_cache = np.reshape(test_y[j, :], [target_size, 1])

        _, _, loss_train_cache, acc_test_cache = f_feedfordward(layer_nn, data_cache, target_cache)

        acc_test += np.divide(acc_test_cache, test_size)

    return loss_train_hist, loss_val_hist, acc_train_hist, acc_val_hist, acc_test

if __name__ == "__main__":

    # Random seed
    np.random.seed(0)

    # NN parameters
    learn_rate_iter = (1e-1, 5e-2, 1e-2)
    activation_iter = ('relu', 'tanh')
    max_epoch = 50
    batch_size = 280

    for learn_rate in learn_rate_iter:
        for activation in activation_iter:
            loss_train_hist, loss_val_hist, acc_train_hist, acc_val_hist, acc_test = d_nn(learn_rate, activation, max_epoch, batch_size, verbose = True)

            print('Accuracy for NN with lr {} and activation {}: {}'.format(learn_rate, activation, acc_test))

            epochs = range(len(acc_train_hist))

            type = 'Training'
            parameter = 'Loss'
            f_plot(epochs, loss_train_hist, type, parameter, learn_rate, activation)

            type = 'Validation'
            parameter = 'Loss'
            f_plot(epochs, loss_val_hist, type, parameter, learn_rate, activation)

            type = 'Training'
            parameter = 'Accuracy'
            f_plot(epochs, acc_train_hist, type, parameter, learn_rate, activation)

            type = 'Validation'
            parameter = 'Accuracy'
            f_plot(epochs, acc_val_hist, type, parameter, learn_rate, activation)