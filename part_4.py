import numpy
import random
import utils
from sklearn.neural_network import MLPClassifier
from numpy.random import permutation
from matplotlib import pyplot as figure


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


# -- preprocessing starts

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


# -- preprocessing is done

clf_1 = MLPClassifier(solver='sgd', hidden_layer_sizes = (16, 128, 64, 32, 16), batch_size=500,
                      activation='relu', alpha=0, momentum=0.0, learning_rate_init=0.1, verbose = True)
clf_01 = MLPClassifier(solver='sgd', hidden_layer_sizes = (16, 128, 64, 32, 16), batch_size=500,
                       activation='relu', alpha=0, momentum=0.0, learning_rate_init=0.01, verbose = True)
clf_001 = MLPClassifier(solver='sgd', hidden_layer_sizes = (16, 128, 64, 32, 16), batch_size=500,
                        activation='relu', alpha=0, momentum=0.0, learning_rate_init=0.001, verbose = True)

# get data_size, define epoch and batch sizes
data_size = len(x_training_set)
batch_size = 500
epoch_size = 10
step = 0
epoch = 0

training_loss_1 = []
training_loss_01 = []
training_loss_001 = []
val_acc_1 = []
val_acc_01 = []
val_acc_001 = []

for epoch in range(epoch_size):
    training_interval = permutation(len(x_training_set))
    x_training_set = x_training_set[training_interval]
    y_training_set = y_training_set[training_interval]

    for i in range(0,data_size,batch_size):
        clf_1.partial_fit(x_training_set[i:i+batch_size], y_training_set[i:i+batch_size],
                          numpy.unique(y_training_set))
        clf_01.partial_fit(x_training_set[i:i+batch_size], y_training_set[i:i+batch_size],
                           numpy.unique(y_training_set))
        clf_001.partial_fit(x_training_set[i:i+batch_size], y_training_set[i:i+batch_size],
                            numpy.unique(y_training_set))

        # increase step and take data in each 10 batch
        step += 1
        if step == 10:
            training_loss_1.append(clf_1.loss_)
            training_loss_01.append(clf_01.loss_)
            training_loss_001.append(clf_001.loss_)
            val_acc_1.append(clf_1.score(x_validation_set,y_validation_set))
            val_acc_01.append(clf_01.score(x_validation_set,y_validation_set))
            val_acc_001.append(clf_001.score(x_validation_set,y_validation_set))
            step = 0

# create dictionary of the selected architecture 5
part4_dict = {
    'name': "arch5",
    'loss_curve_1' : training_loss_1,
    'loss_curve_01' : training_loss_01,
    'loss_curve_001' : training_loss_001,
    'val_acc_curve_1' : val_acc_1,
    'val_acc_curve_01' : val_acc_01,
    'val_acc_curve_001' : val_acc_001
}


utils.part4Plots(part4_dict, "C:/Users/kadir/Desktop/EE496/HW1", "part4", False)

clf_new = MLPClassifier(solver='sgd', hidden_layer_sizes = (16, 128, 64, 32, 16), batch_size=500, activation='relu',
                        alpha=0, momentum=0.0, learning_rate_init=0.01, verbose = True)
val_acc_new = []
test_accuracy = []
while epoch < epoch_size:
    training_interval = permutation(len(x_training_set))
    x_training_set = x_training_set[training_interval]
    y_training_set = y_training_set[training_interval]

    for i in range(0,data_size,batch_size):
        clf_new.partial_fit(x_training_set[i:i+batch_size], y_training_set[i:i+batch_size],
                            numpy.unique(y_training_set))

        # increase step and take data in each 10 batch
        step += 1
        if step == 10:
            val_acc_new.append(clf_new.score(x_validation_set,y_validation_set))
            step = 0
    epoch += 1
    test_accuracy.append(clf_new.score(test_images, test_labels))
    if epoch == 10:
        # set new learning rate and train until 200 epoch
        clf_new.set_params(learning_rate_init=0.001)
        epoch_size = 200

# now plot validation accuracy curve as step 5 & 7 in the document
# caution: comment the line 138 in order for getting the plot properly
val_acc_new = numpy.asarray(val_acc_new)
figure.plot(numpy.arange(len(val_acc_new)),val_acc_new)
figure.xlabel('Step')
figure.ylabel('Validation Accuracy')
figure.title('Validation Accuracy with Learning Rate 0.001')
figure.savefig("C:/Users/kadir/Desktop/EE496/HW1/Validation_Acc_0001")

# finally get test score and print it for comparison with 2.1
test_score = max(test_accuracy)
print("test score", test_score)