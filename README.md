# fortran-curveFitting-NN
Fortran curve fitting neural network program.

This program reads a file with an arbitrary number of points (x,y) which are the training set. These represent point in space of a given function to learn. 

The program then creates a simple 1x24 neural network that iterates over these point, reducing an error function and learning to approximate this data by itself. 

$y = x + 3$

After that, a graph is shown with the results. 
