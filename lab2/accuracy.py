##########################################################################
#      Name  : Divay Gupta      |     Email : gupta576@purdue.edu        #
##########################################################################

################################ Imports #################################
                                                                         #
import matplotlib.pyplot as plt                                          #
import numpy as np                                                       #
import tensorflow as tf                                                  #
from tensorflow import keras                                             #
from tensorflow.keras.utils import to_categorical                        #
import random                                                            #
                                                                         #
##########################################################################



def main():
    ann_accs = [95.20, 82.58, 10.01, 05.02, 01.00]
    cnn_accs = [99.35, 92.81, 74.59, 52.96, 39.71]
    labels   = ["MNIST_d", "MNIST_f", "CIFAR_10", "CIFAR_100_C", "CIFAR_100_F"]
    plt.bar()



if __name__ == '__main__':
    main()

##########################################################################