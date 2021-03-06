##########################################################################
#      Name  : Divay Gupta      |     Email : gupta576@purdue.edu        #
##########################################################################

################################ Imports #################################
                                                                         #
import os                                                                #
import numpy as np                                                       #
import tensorflow as tf                                                  #
from tensorflow import keras                                             #
from tensorflow.keras.utils import to_categorical                        #
import random                                                            #
                                                                         #
##########################################################################

######################### Basic Initialization ###########################
                                                                         #
# Setting random seeds to keep everything deterministic.                 #
random.seed(1618)                                                        #
np.random.seed(1618)                                                     #
tf.random.set_seed(1618)                                                 #
#tf.set_random_seed(1618)   # Uncomment for TF1.                         #
                                                                         #
# Disable some troublesome logging.                                      #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'                                 #
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                                #
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.       #
                                                                         #
##########################################################################

############################ TF Hyperparameters ##########################
                                                                         #
TF_EPOCHS   = 10                                                         #
TF_LR       = 0.001                                                      #
TF_DROP_OUT = 0.20                                                       #
                                                                         #
##########################################################################

########################### Global Constants #############################
                                                                         #
ALGORITHM = "guesser"                                                    #
#ALGORITHM = "tf_net"                                                    #
#ALGORITHM = "tf_conv"                                                   #
                                                                         #
DATASET = "mnist_d"                                                      #
#DATASET = "mnist_f"                                                     #
#DATASET = "cifar_10"                                                    #
#DATASET = "cifar_100_f"                                                 #
#DATASET = "cifar_100_c"                                                 #
                                                                         #
if DATASET == "mnist_d":                                                 #
    NUM_CLASSES = 10                                                     #
    IH = 28                                                              #
    IW = 28                                                              #
    IZ = 1                                                               #
elif DATASET == "mnist_f":                                               #
    NUM_CLASSES = 10                                                     #
    IH = 28                                                              #
    IW = 28                                                              #
    IZ = 1                                                               #
elif DATASET == "cifar_10":                                              #
    NUM_CLASSES = 10                                                     #
    IH = 32                                                              #
    IW = 32                                                              #
    IZ = 3                                                               #
elif DATASET == "cifar_100_f":                                           #
    NUM_CLASSES = 100                                                    #
    IH = 32                                                              #
    IW = 32                                                              #
    IZ = 3                                                               #
elif DATASET == "cifar_100_c":                                           #
    NUM_CLASSES = 20                                                     #
    IH = 32                                                              #
    IW = 32                                                              #
    IZ = 3                                                               #
else:                                                                    #
    raise ValueError("Dataset does not exist or not available")          #
IS = IH * IW * IZ                                                        #
                                                                         #
##########################################################################

########################## Classifier Functions ##########################

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(xTrain, yTrain, eps=TF_EPOCHS, lr=TF_LR):
    print("Building and training TF_NN.")
        
    # Initialize Keras sequential model
    model = keras.Sequential()

    # Add a flattening layer to the model
    model.add(keras.layers.Flatten())

    # Add a dense layer to the model
    model.add(keras.layers.Dense(512, activation=tf.nn.selu))

    # Add an output layer to the model
    model.add(keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax))

    # Initialize model optimizer function
    optim = tf.optimizers.Adam(learning_rate=lr)

    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

    # Train model
    model.fit(xTrain, yTrain, epochs=TF_EPOCHS)
    print("\n\n")

    return model



def buildTFConvNet(xTrain, yTrain, eps=TF_EPOCHS, lr=TF_LR, dropout=True, dropRate=TF_DROP_OUT):
    print("Building and training TF_CNN.")
        
    # Initialize Keras sequential model
    model = keras.Sequential()

    # Add convolutional layers to the model
    model.add(keras.layers.Conv2D(32, kernel_size=[3, 3], activation=tf.nn.relu, input_shape=[IH, IW, IZ]))
    model.add(keras.layers.Conv2D(32, kernel_size=[3, 3], activation=tf.nn.relu))

    # Add pooling and normalization layers to the model
    model.add(keras.layers.MaxPooling2D(pool_size=[2, 2]))
    model.add(keras.layers.BatchNormalization())

    # Add convolutional layers to the model
    model.add(keras.layers.Conv2D(32, kernel_size=[3, 3], activation=tf.nn.relu, input_shape=[IH, IW, IZ]))
    model.add(keras.layers.Conv2D(32, kernel_size=[3, 3], activation=tf.nn.relu))

    # Add pooling and normalization layers to the model
    model.add(keras.layers.MaxPooling2D(pool_size=[2, 2]))
    model.add(keras.layers.BatchNormalization())

    # Add a flattening layer to the model
    model.add(keras.layers.Flatten())

    # Add a dense layer to the model
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))

    # Dropout handling
    if (dropout):
        # Add a dropout layer to the model
        model.add(keras.layers.Dropout(dropRate, input_shape=[2]))

    # Add a dense layer to the model
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))

    # Dropout handling
    if (dropout):
        # Add a dropout layer to the model
        model.add(keras.layers.Dropout(dropRate, input_shape=[2]))

    # Add an output layer to the model
    model.add(keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax))

    # Initialize model optimizer function
    optim = tf.optimizers.Adam(learning_rate=lr)

    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

    # Train model
    model.fit(xTrain, yTrain, epochs=TF_EPOCHS)
    print("\n\n")

    return model

##########################################################################

########################### Pipeline Functions ###########################

def getRawData():
    # MNIST Digit Dataset
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    # MNIST Fashion Dataset
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    # CFAR10 Dataset
    elif DATASET == "cifar_10":
        cifar_10 = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar_10.load_data()
    # CFAR100 Dataset (Fine)
    elif DATASET == "cifar_100_f":
        cifar_100_f = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar_100_f.load_data(label_mode = "fine")
    # CFAR100 Dataset (Coarse)
    elif DATASET == "cifar_100_c":
        cifar_100_c = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar_100_c.load_data(label_mode = "coarse")
    # Error
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s.\n" % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    # Unpack data
    xTest, yTest = data
    yTest = np.argmax(yTest, axis = 1)
    preds = np.argmax(preds, axis = 1)

    # Calculate accuracy
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]

    # Initialize confusion matrix and update values
    confMat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype = np.int32)
    for i in range(preds.shape[0]):
        confMat[yTest[i]][preds[i]] += 1

    # Calculate confusion matrix sum
    confMatSum = np.sum(confMat)

    # Initialize F1 scores vector and update values
    f1 = np.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        # Calculate TPs (True Positives)
        tp = confMat[i][i]

        # Calculate FPs (False Positives)
        fp = sum(confMat[i][j] for j in range(NUM_CLASSES) if (i != j))

        # Calculate FNs (False Negatives)
        fn = sum(confMat[j][i] for j in range(NUM_CLASSES) if (i != j))

        # Calculate F1 Score For Class Label
        f1[i] = float(tp / (tp + (0.5 * (fp + fn))))

    # Display classifier information
    print("Classifier algorithm: %s" % (ALGORITHM))
    print("Classifier accuracy: %f%%" % (accuracy * 100))

    ######################### Confusion Matrix #########################

    # Print confusion matrix title
    print("\n" + "#" * 28 + " Confusion Matrix " + "#" * 28)

    # Print column labels
    print(" " * 3, end = "")
    for i in range(NUM_CLASSES):
        print("%6d" % (i), end = " ")
    print("\n  ???" + "???" * (7*NUM_CLASSES) + "???" , end = "")
    print()

    # Print confusion matrix rows
    for i in range(len(confMat)):
        print("%d ???" % i, end = " ")
        for val in confMat[i]:
            print("{:5d}".format(val), end=" ")
            print("???", end="")
        if i != len(confMat) - 1:
            print()
    print("\n  ???" + "???" * (7*NUM_CLASSES) + "???")

    ####################################################################

    ########################### F1 Score Table #########################

    # Print F1 score title
    print("\n" + "#" * 30 + " F1 Scores " + "#" * 30)
    
    # Print column labels
    for i in range(NUM_CLASSES):
        print("%7d" % (i), end = "")
    print()

    # Print F1 Scores
    print(np.around(f1, decimals = 4))

    ####################################################################

##########################################################################

################################## Main ##################################

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()

##########################################################################