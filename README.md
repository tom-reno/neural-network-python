# Neural Network Python

A simple neural network in Python without libraries like TensorFlow or PyTorch to train and predict characters on images
or CSV datasets. Because these CSV datasets are usually too large to put on GitHub, you can find a training set, for 
example, at [kaggle](https://www.kaggle.com/datasets?search=character+recognition). You can find a test CSV dataset in
`./data/training/csv` and images in `./data/training/img`, though.

## Requirements

- Python >= 3.13
- uv >= 0.8.11

## Installation

```
cd <workspace>

git clone git@github.com:tom-reno/neural-network-python.git

cd neural-network-python

uv sync
```

## Run

```
uv run neural_network <optional arguments, e.g. --amount-hidden-layers=2>
```

| Property                     | Argument                       | Default                                                                                                                                                             | Description                                                  |
|------------------------------|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| --mode                       | prediction, training           | training                                                                                                                                                            | Defines the network mode                                     |
| --datatype                   | csv, img                       | csv                                                                                                                                                                 | Defines which data type the files are to be trained          |
| --data                       | any string                     | - ./data/prediction for --mode=prediction,<br>- ./data/training/csv for --mode=training and --type=csv,<br>- ./data/training/img for --mode=training and --type=img | Defines the location of the files to be predicted or trained |
| --amount-hidden-layers       | any integer number             | 2                                                                                                                                                                   | Defines the amount of hidden layers                          |
| --hidden-activation-function | sigmoid[, softmax, tanh, relu] | sigmoid                                                                                                                                                             | Defines the activation function for the hidden layers        |
| --output-activation-function | sigmoid[, softmax, tanh, relu] | sigmoid                                                                                                                                                             | Defines the activation function for the output layer         |
| --learning-rate              | any floating point number      | 0.01                                                                                                                                                                | Defines the learning rate                                    |
| --training-iterations        | any integer number             | 1000                                                                                                                                                                | Defines the amount of training iterations                    |
| --batch-size                 | any integer number             | 10                                                                                                                                                                  | Defines the size of batches to be trained in at once         |
