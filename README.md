# CESN
A new neural network architecture for time series classification and regression. 

# CESN architecture
Here we present and describe in detail a new neural network architecture called CESN, Convolutional Echo State Network, which results from the combination of the Convolutional Neural Networks (CNNs) and the Echo State Networks (ESNs). CESN results to be appropriate for Time Series Classification (TSC) tasks and Time Series Regression (TSR) tasks, just changing the output activation function and the loss function of the model. 
CESN architecture in details is shown in the figure below 

<img src="images/cesn.png" alt="cesn_architecture" width="650"/>

* **Input layer**: It brings the input data, that can be univariate or multivariate timeseries, into the next layer.

* **Reservoir layer**: This layer aims to project the input data into a high dimensional space, trough temporal and non-linear activation functions of each neuron into the reservoir. This dimensional augmentation allows visualizing more patterns than the data in the original dimension. 

![](https://latex.codecogs.com/svg.image?\vec{z}(t&plus;1)&space;=&space;f(W^{in}x(t&plus;1)&plus;W^{res}z(t))&space;\in&space;\mathbb{R}^M)

* **Convolutional layer**: Same as the CNNs this layer consists of a convolutional and a pooling phase in which the features of the reservoir state are extracted in the same way performed  in  image classification.
* **Flattening layer**: In this layer approaching data from the convolutional layer are flatted to be fed into the next layer. This means that data coming from the convolutional layer, which is a tensor with rank 3 are mapped into a 1-D array, a vector, in order to be feed into the fully connected layer.
* **Fully Connected layer**: This is just a simple deep forward neural network used for the classification of time series transformed by the previous layer into a specific feature.

# CESN - Classification

CESN can be used for time series classification tasks. Here we present some examples how to use it for the classification task. 
We consider three different dataset, typically used in this context:
* SisFall dataser - [https://github.com/your_username/repo_name](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5298771/)
* ECG200 - [https://timeseriesclassification.com/description.php?Dataset=ECG200](https://timeseriesclassification.com/description.php?Dataset=ECG200)
* ECG5000 - [https://timeseriesclassification.com/description.php?Dataset=ECG5000](https://timeseriesclassification.com/description.php?Dataset=ECG5000)

Here you can find the ECG200 and ECG500 dataser, while the SisFall dataset is avalaible at the following link

In the folder /
