# SVM-Sequential-Minimal-Optimization-From-Scratch

This project was made for the course "Neural Networks & Deep Learning". Its task was to build a **Support Vector Machine** with linear and radial kernel from scratch,
use it to classify images and compare its performance to the **K-Nearest Neighbors** and **Nearest Centroid** algorithms.


The SVM solves the dual quadadratic programming problem with the **Sequential Minimal Optimization** [[1]](#1) (SMO) algorithm. It supports 
linear, polynomial and radial kernels and uses an error cache for faster convergence.

It was trained and evaluated with the [MNIST Fashion](https://github.com/zalandoresearch/fashion-mnist) dataset for binary classification, by choosing 2 out of the 10 
classes available. In the [main.ipynb](main.ipynb) it is trained for two different class pairs. The results are compared to **KNN** and **NC** models for the first pair
of classes and to [libsvm](https://github.com/cjlin1/libsvm)'s SVM, which also uses SMO, for the second pair.

A more detailed report is available here [Neural_Networks_SVM.pdf](https://github.com/Kouspan/SVM-Sequential-Minimal-Optimization-From-Scratch/files/8901514/Neural_Networks__SVM.pdf)
(in greek).

### Comments for running the notebook. 
With the process_data(...) function the dataset is loaded from the ubyte files and processed with PCA and OneHotEncoder.
For ease of use, the processed data are saved in .npy form in the './Processed/' folder.
Either create the above folder or change the save path.
Also the grid searches last around 1 hour for the linear kernel and 3+ hours for the radial. Their results are saved in the .json files, run the cell bellow each grid
search to load the json files and plot the results.









## References
<a id="1">[1]</a>
Platt, John. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines.
Advances in Kernel Methods-Support Vector Learning, 208


