# MML
Machine Learning Library

## Examples of usage

Simple Feed Forward Neural Network:
```cpp
#include "Model.h"
#include "Layers/Dense.h"
#include "Layers/Activation.h"

int main() {
    // Initialization of training data and labels
    ml::Data trainingData({
        ml::Vector({0.0, 0.5, 1.0}), // first example of data set
        ml::Vector({1.0, 0.0, 0.1}), // second one
    });
    ml::Data labels({
        ml::Vector({1, 0}), // a label to fisrt data set element
        ml::Vector({0, 1}), // and for a second one
    });

    // Creation of model
    ml::Model model({
        new ml::Dense(3, 4),
        new ml::Activation("leaky ReLU"),
        new ml::Dense(4, 2),
        new ml::Activation("sigmoid"),
    });
    model.compile("SGD", "MSE"); // set SGD optimzer and MSE loss function for the model

    // Training the model
    model.train(trainingData, labels, 100); // epoches = 100

    ml::Tensor output = model.inference(trainingData[0]);

    return 0;
}
```

Simple Convolutional Neural Network:
```cpp
#include "Model.h"
#include "Layers/Convolutional.h"
#include "Layers/Pooling.h"
#include "Layers/Flatten.h"
#include "Layers/Dense.h"
#include "Layers/Activation.h"

int main() {
    // Initialization of training data and labels
    ml::Data trainingData({
        ml::Tensor({1, 6, 6}, { // first example of data set with size = {6, 6} and depth = 1
            0, 1, 1, 1, 1, 0,
            1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1,
            0, 1, 1, 1, 1, 0,
}),
        ml::Tensor({1, 6, 6}, { // second example
            1, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 0, 0,
            0, 0, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            1, 0, 0, 0, 0, 1,
}),
    });

    ml::Data labels({
        ml::Vector({1, 0}), // a label to fisrt data set element
        ml::Vector({0, 1}), // and for a second one
    });

    // Creation of model
    ml::Model model({
        new ml::Convolutional({1, 6, 6}, {3, 3}, 4), // input shape = {1, 6, 6} , kernel shape = {3, 3} , depth = 4
        new ml::Pooling(2, 2), // strides = 2, pooling size = 2
        new ml::Activation("ReLU"),
        new ml::Flatten(),
        new ml::Dense(16, 2),
        new ml::Activation("sigmoid"),
    });
    model.compile("SGD", "MSE"); // set SGD optimzer and MSE loss function for the model

    // Training of model
    model.train(trainingData, labels, 100); // epoches

    ml::Tensor output = model.inference(trainingData[0]);

    return 0;
}
```
