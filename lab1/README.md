# Lab1

#### Completed Lab Sections:
*	Custom Neural Net
--	Accuracy – 93.23%
--	Implemented the sigmoid and sigmoid derivative functions.
--	Implemented the train function using backpropagation properly.
--	Fully functioning 2-layer neural net with minibatches of size 100.
*	TensorFlow Neural Net
--	Accuracy – 98.27%
--	Implemented a 2-layer neural net using Keras
--	Used the sigmoid and SoftMax activation functions, in that order
--	Used the Adam optimizer
*	Pipeline
--	Normalized image data values from 0--255 to a range of 0.0--1.0
--	Implemented a F1 score table and a confusion matrix with accuracy values

#### Resources Used:
*	Lecture Slides
--	Slides 2 – Neural Network and Backpropagation basics
--	Slides 3 – Backpropagation and TensorFlow/Keras
--	Slides 4 – Tuning hyperparameters
•	API Documentation Links
--	TensorFlow
--	Keras
•	Miscellaneous Links
--	Backpropagation using Python
--	One Hot Encoding
--	MNIST Image Classification using Keras
--	What is a Confusion Matrix in Machine Learning? 


#### Lab Implementation Summary:
*	Guesser Algorithm
--	This algorithm just randomly guesses the labels with a set seed of 1618 for deterministic behavior and it achieves an understandably low accuracy of 9.68%.

*	Custom Neural Net
--	The custom neural net was implemented with 2 layers – one hidden layer and an output layer. The sigmoid function used was implemented and used as the activation function. Backpropagation was correctly implemented and the outputs of the network are then one-hot encoded to get the predictions.
--	The neural net was trained for 30 epochs at a learning rate of 0.01 with a 100 mini-batches and was able to achieve an accuracy of 93.23%.

*	TensorFlow Neural Net
--	This neural net was implemented using Keras, with two layers – one hidden layer and an output layer. The hidden layer has 512 neurons (same as the input image size) and used the sigmoid function as an activation function and the output layer has 10 neurons (same as number of classifiers) and used the SoftMax function as its activation function. Those two activation functions were chosen since that combination gave the lowest loss and hence the best accuracy. The Adam function was used for gradient descent and the categorical cross entropy function was defined as the loss function due to its high performance in categorical data classification. The outputs of the network are then one-hot encoded to get the predictions.
--	The neural net was trained for 20 epochs at a learning rate of 0.001 and was able to achieve an accuracy of 98.27%.
