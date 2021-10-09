# Lab2

#### Resources Used:
*	Lecture Slides
```
	Slides 5 – Convolutional Neural Networks, Pooling, Optimization
	Slides 6 – Convolutional Neural Networks 2, CNNS using TensorFlow and Keras
```

*	API Documentation Links
```
	TensorFlow
	Keras
```

#### Lab Implementation Summary:
*	TensorFlow ANN (tf_net)
Implemented using TensorFlow and Keras; fully functional
```
	Accuracy:

		MNIST_d		: 	95.20%	
		MNIST_f		: 	82.58%
		CIFAR_10	: 	10.01%
		CIFAR_100_C	: 	05.02%
		CIFAR_100_F	: 	01.00%

```

*	TensorFlow CNN (tf_conv)
Implemented using TensorFlow and Keras 
Fully functional
```
	Accuracy:
		MNIST_d		: 	99.35%
		MNIST_f		: 	92.81%
		CIFAR_10	: 	74.59%
		CIFAR_100_C	: 	52.96%
		CIFAR_100_F	: 	39.71%
```

*	Pipeline
Can use the TensorFlow ANN by setting ALGORITHM to “tf_net”
Can use the TensorFlow CNN by setting ALGORITHM to “tf_conv”
```	
Can train the neural nets on the following datasets:
		MNIST_d 
		MNIST_f 
		CIFAR_10
		CIFAR_100_C
		CIFAR_100_F
```