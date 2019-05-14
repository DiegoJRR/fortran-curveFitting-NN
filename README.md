# fortran-curveFitting-NN
Fortran curve fitting neural network program.

This program reads a file with an arbitrary number of points (x,y) which are the training set. These represent point in space of a given function to learn. 

The program then creates a simple 1x24 (1 hidden layer with 24 neurons) neural network that iterates over these point, reducing an error function and learning to approximate this data by itself. 

![1200px-Colored_neural_network svg](https://user-images.githubusercontent.com/29646853/57186207-4ac8b100-6ea0-11e9-908f-be5d64bbcad8.png)

The neural network is a feed forward network minimazing the loss function:

![Capture](https://user-images.githubusercontent.com/29646853/57186214-8794a800-6ea0-11e9-8fa2-9a0405ad35db.PNG)

using backpropagation.

After that, a graph is shown with the results.
