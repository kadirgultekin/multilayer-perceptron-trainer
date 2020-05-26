import numpy
import random
from sklearn.neural_network import MLPClassifier
import utils
from numpy.random import permutation


train_images = numpy.load('train_images.npy')
train_labels = numpy.load('train_labels.npy')
test_images = numpy.load('test_images.npy')
test_labels = numpy.load('test_labels.npy')


# this helper function does the random distribution of the data
# as the 10% of the data is splitted for the validation and 
# the rest is for training
def generate_data(arr):
    # generate 600 unique(distinct) random numbers between 0 and 6000
    random_numbers = random.sample(range(0, 6000), 600)
    # create validation array with indexes of generated random numbers
    validation = [arr[i] for i in random_numbers]
    # create training array with remaining data from the validation array
    training = [arr[i] for i in range(0,6000) if i not in random_numbers]
    return training, validation


# this helper method generates plots using utils.py
def generate_plots():
    # plot performances
    utils.part2Plots(dict_lists, "C:/Users/kadir/Desktop/EE496/HW1", "part2", False)
    # plot weights
    # arch1 is excluded
    dict_lists_ex = dict_lists[1:]
    counter = 1 
    for i in dict_lists_ex:
        utils.visualizeWeights(i['weights'],'C:/Users/kadir/Desktop/EE496/HW1',
                     filename=archs[counter])
        counter+=1


# Scale pixel values to [-1.0,1.0]
train_images = train_images.astype(numpy.float)
train_images = ((2*train_images)/255)-1

test_images = test_images.astype(numpy.float)
test_images = ((2*test_images)/255)-1

# Zip (match) each class with its corresponding label
# Then sort each class according to its label
train_data = zip(train_labels, train_images)
train_data_sorted = sorted(train_data, key=lambda data: data[0])

# Remove label from the sorted data, leave it with pure sorted data
train_data_pure = [data[1] for data in train_data_sorted]

# Generate classes
class_0 = train_data_pure[:6000]
class_1 = train_data_pure[6000:12000]
class_2 = train_data_pure[12000:18000]
class_3 = train_data_pure[18000:24000]
class_4 = train_data_pure[24000:]

# Generate training and validation lists randomly by using helper function
training_0,validation_0 = generate_data(class_0)
training_1,validation_1 = generate_data(class_1)
training_2,validation_2 = generate_data(class_2)
training_3,validation_3 = generate_data(class_3)
training_4,validation_4 = generate_data(class_4)

# Concatenate training and validation sets
x_training_set = training_0 + training_1 + training_2 + training_3 + training_4
x_validation_set = validation_0 + validation_1 + validation_2 + validation_3 + validation_4

# Convert lists to numpy arrays
x_training_set = numpy.asarray(x_training_set)
x_validation_set = numpy.asarray(x_validation_set)

# Now create expected outputs.
# Since we have sorted data, our work for expected output is easier
# Just append 5400+600 for class 0, 5400+600 for class 1 etc. 
y_training_set = numpy.array([])
y_validation_set  = numpy.array([])
for i in range(5):
    for j in range(5400):  # 27000/5 = 5400
        y_training_set = numpy.append(y_training_set, i)
    for k in range(600):  # 3000/5 = 600
        y_validation_set = numpy.append(y_validation_set, i)

# get data_size, define epoch and batch sizes
data_size = len(x_training_set)  # 27000
batch_size = 500
epoch_size = 100

# in order to train all architectures at ones, create a list of architectures
# with a list of tuples that contain hidden layers
archs = ['arch1', 'arch2', 'arch3', 'arch5', 'arch7']
hidden_layers = [128, (16, 128), (16, 128, 16), (16, 128, 64, 32, 16), (16, 32, 64, 128, 64, 32, 16)]
dict_lists = [] # global list to hold dictionaries to be used in utils.py

# -- training starts --
for idx in range(len(hidden_layers)):
    # define all necessary variables and lists
    step = 0
    number_of_iter = 10
    training_loss = []
    training_accuracy = []
    validation_accuracy = []
    test_accuracy = []
    coefs = []

    for iteration in range(number_of_iter):
        clf = MLPClassifier(solver="adam", hidden_layer_sizes=hidden_layers[idx],
                            batch_size=batch_size, activation="relu", verbose=True)
        # create temp lists to hold intermediate results
        train_loss_temp = []
        train_acc_temp = []
        val_acc_temp = []

        for epoch in range(epoch_size):
            training_interval = permutation(len(x_training_set))
            x_training_set = x_training_set[training_interval]
            y_training_set = y_training_set[training_interval]

            for i in range(0,data_size,batch_size): # 0 - 27000 - 500
                clf.partial_fit(x_training_set[i:i+batch_size], y_training_set[i:i+batch_size],
                                numpy.unique(y_training_set))
                # increase step and take data in each 10 batch
                step+=1
                if step == 10:
                    train_loss_temp.append(clf.loss_)
                    train_acc_temp.append(clf.score(x_training_set, y_training_set))
                    val_acc_temp.append(clf.score(x_validation_set, y_validation_set))
                    step = 0
        # training loss, training accuracy, and validation accuracy are list of lists
        training_loss.append(train_loss_temp)
        training_accuracy.append(train_acc_temp)
        validation_accuracy.append(val_acc_temp)
        test_accuracy.append(clf.score(test_images, test_labels))
        coefs.append(clf.coefs_[0])
    # -- end of training --

    # get best test accuracy and the best weight
    test_accuracy = [max(test_accuracy)]
    weight = coefs[test_accuracy.index(max(test_accuracy))]

    # sum list of lists by using list comprehension and zip 
    # zip returns a pointer to the list, use * to get the value 
    training_loss_sum = [sum(i) for i in zip(*training_loss)]
    training_accuracy_sum = [sum(i) for i in zip(*training_accuracy)]
    validation_accuracy_sum = [sum(i) for i in zip(*validation_accuracy)]

    # get averages as the final work
    training_loss_final = [i/number_of_iter for i in training_loss_sum]
    training_accuracy_final = [i/number_of_iter for i in training_accuracy_sum]
    validation_accuracy_final = [i/number_of_iter for i in validation_accuracy_sum]

    # create dictionary of the architecture
    part2_dict = {
        'name': archs[idx],
        'loss_curve':  training_loss_final,
        'train_acc_curve': training_accuracy_final,
        'val_acc_curve': validation_accuracy_final,
        'test_acc': test_accuracy,
        'weights': weight
    }

    # append dictionary to global dictionary list
    dict_lists.append(part2_dict)

# as a last job, call helper method to generate plots by making use of the utils.py
generate_plots()