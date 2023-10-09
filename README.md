# MML
Machine Learning Library

## Example of usage
```cpp
#include "Model.h"
#include "Layers/Dense.h"
#include "Layers/Activation.h"

int main() {
    // Initialization of training data and labels
    ml::Data trainingData({ /* ... */ });
    ml::Data labels({ /* ... */ });

    // Creation of model
    ml::Model model({
        new ml::Dense(3, 4),
        new ml::Activation("leaky ReLU"),
        new ml::Dense(4, 2),
        new ml::Activation("sigmoid"),
    });

    // Training of model
    model.train(trainingData, labels, 100); // epoches

    ml::Tensor output = model(trainingData[0]);

    return 0;
}
```
