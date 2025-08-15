# Neural Network Python

A simple neural network in Python.

## Requirements

- Python >= 3.13

## Installation

```
cd <workspace>

git clone git@github.com:tom-reno/neural-network-python.git

cd neural-network-python

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -r requirements.txt
```

## Run

```
python3 -m neural_network <optional arguments, e.g. --mode=training>
```

| Argument                     | Choises                        | Default       | Description                                           |
|------------------------------|--------------------------------|---------------|-------------------------------------------------------|
| --mode                       | evaluation, training           | training      | Defines the network mode                              |
| --data                       | any string                     | ./data/{mode} | Defines the location of the images                    |
| --amount-hidden-layers       | any integer number             | 1             | Defines the amount of hidden layers                   |
| --hidden-activation-function | sigmoid[, softmax, tanh, relu] | sigmoid       | Defines the activation function for the hidden layers |
| --output-activation-function | sigmoid[, softmax, tanh, relu] | sigmoid       | Defines the activation function for the output layer  |
| --learning-rate              | any floating point number      | 0.01          | Defines the learning rate                             |
| --training-iterations        | any integer number             | 1000          | Defines the amount of training iterations             |
