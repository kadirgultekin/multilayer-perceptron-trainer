import numpy
import utils
from sklearn.neural_network import MLPClassifier
from numpy.random import permutation
from numpy.linalg import norm


# helper method to create dictionary
def create_dict(arch_name,training_loss_relu,training_loss_sig,grad_relu,grad_sig):
    # first: convert lists to numpy arrays 
    training_loss_relu = numpy.asarray(training_loss_relu)
    training_loss_sig = numpy.asarray(training_loss_sig)
    grad_relu = numpy.asarray(grad_relu)
    grad_sig = numpy.asarray(grad_sig)

    # second: create dictionary of the architecture with the given arguments
    dict_obj = {
        "name": arch_name,
        "relu_loss_curve": training_loss_relu,
        "sigmoid_loss_curve": training_loss_sig,
        "relu_grad_curve": grad_relu,
        "sigmoid_grad_curve": grad_sig
    }
    # final: append dictionary to global dictionary list
    dict_lists.append(dict_obj)


# helper method to generate plots using utils.py
def generate_plots():
    utils.part3Plots(dict_lists,"C:/Users/kadir/Desktop/EE496/HW1","part3",False)


# load numpy arrays
train_images = numpy.load('train_images.npy')
train_labels = numpy.load('train_labels.npy')
test_images = numpy.load('test_images.npy')
test_labels = numpy.load('test_labels.npy')

# scale pixel values to [-1.0,1.0]
train_images = train_images.astype(numpy.float)
train_images = ((2*train_images)/255)-1

test_images = test_images.astype(numpy.float)
test_images = ((2*test_images)/255)-1

# get data_size, define epoch and batch sizes
data_size = len(train_images) # 30000
epoch_size = 100
batch_size = 500

# in order to train all architectures at ones, create a list of architectures
# with a list of tuples that contain hidden layers
archs = ['arch1', 'arch2', 'arch3', 'arch5', 'arch7']
hidden_layers = [128, (16, 128), (16, 128, 16), (16, 128, 64, 32, 16), (16, 32, 64, 128, 64, 32, 16)]
dict_lists = [] # global list to hold dictionaries to be used in utils.py

# -- training starts --
for idx in range(len(hidden_layers)):
    # define variables and lists
    step = 0
    learning_rate = 0.01
    training_loss_relu = []
    training_loss_sig = []
    coefs_relu = numpy.zeros((784,128))
    coefs_sig = numpy.zeros((784,128))
    grad_relu = []
    grad_sig = []
    first_init = True # will be used for gradient computations
    
    # classifier definitions
    clf_relu = MLPClassifier(solver="sgd", hidden_layer_sizes=hidden_layers[idx],
                             batch_size=batch_size, activation="relu", learning_rate_init=learning_rate,
                             alpha=0, momentum=0.0, verbose=True)
    clf_sig = MLPClassifier(solver="sgd", hidden_layer_sizes=hidden_layers[idx],
                            batch_size=batch_size, activation="logistic",
                            learning_rate_init=learning_rate, alpha=0, momentum=0.0, verbose=True)
    
    for epoch in range(epoch_size):
        # shuffle both images and labels using permutation
        training_interval = permutation(len(train_images)) 
        train_images = train_images[training_interval]
        train_labels = train_labels[training_interval]

        for i in range(0,data_size,batch_size): # 0 - 30000 - 500
            clf_relu.partial_fit(train_images[i:i+batch_size], train_labels[i:i+batch_size],
                                 numpy.unique(train_labels))
            clf_sig.partial_fit(train_images[i:i+batch_size], train_labels[i:i+batch_size],
                                numpy.unique(train_labels))
            
            if step % 10 == 0:
                # get training losses
                training_loss_relu.append(clf_relu.loss_)
                training_loss_sig.append(clf_sig.loss_)
                # check if first_init true, if so get first prev coefs to calculate gradient in the next step
                # this step is necessary to calculate gradient acc. to formula given in 1.2. in the report
                if first_init:
                    coefs_relu = numpy.copy(clf_relu.coefs_[0])
                    coefs_sig = numpy.copy(clf_sig.coefs_[0])
                    first_init = False
                else:
                    # now calculate gradient
                    grad_relu.append(norm((clf_relu.coefs_[0]-coefs_relu)/learning_rate))
                    grad_sig.append(norm((clf_sig.coefs_[0]-coefs_sig)/learning_rate))

                    # get coefficients
                    coefs_relu = numpy.copy(clf_relu.coefs_[0])
                    coefs_sig = numpy.copy(clf_sig.coefs_[0])
            
            # increase step
            step += 1
    
    # create dictionary
    create_dict(archs[idx], training_loss_relu, training_loss_sig, grad_relu, grad_sig)

    # -- end of training --

# as a last job, call helper method to generate plots by making use of the utils.py
generate_plots()