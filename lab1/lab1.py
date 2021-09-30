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
import tqdm                                                              #
                                                                         #
##########################################################################

########################### Global Constants #############################
                                                                         #
# Information on dataset.                                                #
IMAGE_SIZE  = 784                                                        #
NUM_CLASSES = 10                                                         #
NUM_NEURONS = 512                                                        #
                                                                         #
# Custom neural network instance parameters                              #
NN2L_EPOCHS = 30                                                         #
NN2L_MBS    = 100                                                        #
NN2L_LR     = 0.01                                                       #
NN2L_USE_MB = True                                                       #
                                                                         #
# Use these to set the algorithm to use.                                 #
#ALGORITHM = "guesser"                                                   #
ALGORITHM = "custom_net"                                                #
#ALGORITHM = "tf_net"                                                     #
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                                 #
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.       #
                                                                         #
##########################################################################

###################### Custom Neural Network Class #######################

class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, lr = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = lr
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        ds = self.__sigmoid(x)
        return np.multiply(ds, (1 - ds))

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, m, n):
        for i in range(0, len(l), n):
            yield l[i : i + n], m[i : i + n]

    # MSE Loss calculation.
    def __calculate_mse(self, yPred, yOut):
        loss = 0.5 * np.sum((yPred - yOut)**2) # MSE value
        return loss

    # Basic Loss calculation.
    def __calculate_loss(self, yPred, yOut):
        return np.subtract(yOut, yPred)

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)   # Forward pass
        return layer2

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        # Iterate through epochs
        for i in range(0, epochs):
            # Display progress
            print("\r[+] Epoch {}/{} ".format(i + 1, epochs), end = "")

            if (minibatches):
                # Minibatch training
                for xBatch, yBatch in (self.__batchGenerator(xVals, yVals, mbs)):
                    # Forward pass input batch
                    layer1, layer2 = self.__forward(xBatch)                

                    # Calculate loss
                    lossValue = self.__calculate_loss(yBatch, layer2)       
                    
                    # Calculate activation prime for output
                    dSigmoid = self.__sigmoidDerivative(layer2)

                    # Calculate output layer output differences
                    outDiffs = np.multiply(lossValue, dSigmoid)

                    # Calculate error derivatives
                    dError = np.dot(outDiffs, np.transpose(self.W2))

                    # Calculate activation prime for input batch
                    dSigmoid = self.__sigmoidDerivative(layer1)

                    # Calculate hidden layer output differences
                    hidDiffs = np.multiply(dError, dSigmoid)

                    # Adjust output layer weights
                    outWeights = np.matmul(np.transpose(layer1), outDiffs)
                    self.W2 -= np.multiply(outWeights, self.lr)

                    # Adjust hidden layer weights
                    hidWeights = np.matmul(np.transpose(xBatch), hidDiffs)
                    self.W1 -= np.multiply(hidWeights, self.lr)
            else:
                # Forward pass input
                layer1, layer2 = self.__forward(xVals)

                # Calculate loss
                lossValue = self.__calculate_loss(yVals, layer2)

                # Calculate activation prime for output
                dSigmoid = self.__sigmoidDerivative(layer2)

                # Calculate output layer output differences
                outDiffs = np.multiply(lossValue, dSigmoid)

                # Calculate error derivatives
                dError = np.dot(outDiffs, np.transpose(self.W2))

                # Calculate activation prime for input batch
                dSigmoid = self.__sigmoidDerivative(layer1)

                # Calculate hidden layer output differences
                hidDiffs = np.multiply(dError, dSigmoid)

                # Adjust output layer weights
                outWeights = np.matmul(np.transpose(layer1), outDiffs)
                self.W2 -= np.multiply(outWeights, self.lr)

                # Adjust hidden layer weights
                hidWeights = np.matmul(np.transpose(xVals), hidDiffs)
                self.W1 -= np.multiply(hidWeights, self.lr)

##########################################################################

############################ Utility Functions ###########################

# Normalize predictions to probabilities
def normalize(pred):
    # Initialize array of probabilites
    prob = np.zeros(pred.shape)

    # Get classification information with max probability
    maxProbs = np.argmax(pred, axis = 1)

    # Iterate for each prediction
    for i in range(pred.shape[0]):
        prob[i][maxProbs[i]] = 1

    return prob


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

##########################################################################


########################### Pipeline Functions ###########################

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    
    # Normalize input
    xTrain = np.divide(xTrain, 255, dtype = np.float16)
    xTest = np.divide(xTest, 255, dtype = np.float16)

    # Reshape input
    xTrain = xTrain.reshape(-1, 28 ** 2)
    xTest = xTest.reshape(-1, 28 ** 2)

    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)

    # Display information
    print("\nNew shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s.\n" % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.

    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.\n")
        
        # Initialize custom neural network model
        model = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, NUM_NEURONS, lr = NN2L_LR)

        # Train model
        model.train(xTrain, yTrain, epochs = NN2L_EPOCHS, minibatches = NN2L_USE_MB)

        print("\n\n")

        return model

    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        return None

    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)

    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")

        # Run custom neural network model and get predictions
        pred = model.predict(data)
        pred = normalize(pred)

        return np.array(pred)

    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        return None
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
    print("\n  ┌" + "─" * (7*NUM_CLASSES) + "┐" , end = "")
    print()

    # Print confusion matrix rows
    for i in range(len(confMat)):
        print("%d │" % i, end = " ")
        for val in confMat[i]:
            print("{:5d}".format(val), end=" ")
            print("│", end="")
        if i != len(confMat) - 1:
            print()
    print("\n  └" + "─" * (7*NUM_CLASSES) + "┘")

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
