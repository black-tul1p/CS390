##########################################################################
#      Name  : Divay Gupta      |     Email : gupta576@purdue.edu        #
##########################################################################

################################ Imports #################################
                                                                         #
import matplotlib.pyplot as plt                                          #
                                                                         #
##########################################################################


def main():
    ann_accs = [95.20, 82.58, 10.01, 05.02, 01.00]
    cnn_accs = [99.35, 92.81, 74.59, 52.96, 39.71]
    labels   = ["MNIST_d", "MNIST_f", "CIFAR_10", "CIFAR_100_C", "CIFAR_100_F"]

    NN = input("Enter the Neural Net you want metrics for [ANN / CNN]: ")

    if NN.lower() == "ann":
        plt.bar(labels, ann_accs)
        plt.title("Artificial Neural Network Dataset vs. Accuracy")
        plt.xlabel('Datasets', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0,100)
        plt.show()
    elif NN.lower() == "cnn":
        plt.bar(labels, cnn_accs)
        plt.title("Convolutional Neural Network Dataset vs. Accuracy")
        plt.xlabel('Datasets', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0,100)
        plt.show()
    else:
        raise ValueError("Unrecognized input.")


if __name__ == '__main__':
    main()

##########################################################################