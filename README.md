# Neural Network Python

A simple neural network in Python to evaluate images with numbers.

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

| Property                     | Argument                       | Default       | Description                                           |
|------------------------------|--------------------------------|---------------|-------------------------------------------------------|
| --mode                       | evaluation, training           | training      | Defines the network mode                              |
| --data                       | any string                     | ./data/{mode} | Defines the location of the images                    |
| --amount-hidden-layers       | any integer number             | 1             | Defines the amount of hidden layers                   |
| --hidden-activation-function | sigmoid[, softmax, tanh, relu] | sigmoid       | Defines the activation function for the hidden layers |
| --output-activation-function | sigmoid[, softmax, tanh, relu] | sigmoid       | Defines the activation function for the output layer  |
| --learning-rate              | any floating point number      | 0.01          | Defines the learning rate                             |
| --training-iterations        | any integer number             | 1000          | Defines the amount of training iterations             |
