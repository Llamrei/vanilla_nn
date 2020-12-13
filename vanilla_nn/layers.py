from .functions import Sum
from .functions import Prod
from .functions import Const
from .functions import Var
from .functions import Neg
from .functions import MSE
from .metrics import manhattan

from functools import reduce
from functools import partial
import logging
import sys

from random import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler("debug.log"))


class Layer:
    # Has a list of neurons which are all just Functions
    def __init__(self, num_neurons, weight_init_func=random):
        self.neurons = [(None)] * num_neurons
        self.num_neurons = num_neurons
        self.nodes_namespace = None
        # Weights are defined looking back
        self.weights = dict()
        self.prev_layer = None
        self.weight_init_func = weight_init_func

    def __str__(self):
        # Careful that this code is seperate from usage in the dense_append_to
        return f"{[(x, i+self.nodes_namespace[0]) for i, x in enumerate(self.neurons)]}"

    def dense_append_to(self, previous_layer):
        self.prev_layer = previous_layer
        self.nodes_namespace = (
            previous_layer.nodes_namespace[1] + 1,
            previous_layer.nodes_namespace[1] + self.num_neurons,
        )
        for neuron_name, neuron_idx in enumerate(
            range(self.num_neurons), self.nodes_namespace[0]
        ):
            weighted_previous_layer = []
            for prev_name, prev_neuron in enumerate(
                previous_layer, previous_layer.nodes_namespace[0]
            ):
                weight = f"w{prev_name}_{neuron_name}"
                weighted_previous_layer.append(Prod(prev_neuron, Var(weight)))
                self.weights[weight] = self.weight_init_func()
            self.neurons[neuron_idx] = reduce(Sum, weighted_previous_layer)

    def inputs(self, neurons):
        assert len(neurons) == self.num_neurons

        for idx, neuron in enumerate(neurons):
            self.neurons[idx] = neuron
        self.nodes_namespace = (0, self.num_neurons - 1)

    @property
    def all_layer_weights(self):
        weights = self.weights
        prev_layer = self.prev_layer
        while prev_layer:
            weights = {**weights, **prev_layer.weights}
            prev_layer = prev_layer.prev_layer
        logger.debug(f"Weights retrieved: {weights}")
        return weights

    def set_all_layer_weights(self, weights):
        prev_layer = self
        while prev_layer:
            for key in prev_layer.weights:
                prev_layer.weights[key] = weights[key]
            prev_layer = prev_layer.prev_layer

    # TODO: could be cool to add memoization for some dynamic programming swag, see lru_cache from functools
    def layer_grad(self, at):
        grad = [dict()] * self.num_neurons
        for neuron_idx, _ in enumerate(self.neurons):
            prev_layer = self
            while prev_layer:
                for weight in prev_layer.weights:
                    grad[neuron_idx][weight] = prev_layer.neurons[neuron_idx].diff(
                        weight, at
                    )
                prev_layer = prev_layer.prev_layer
        return grad

    def __iter__(self):
        return iter(self.neurons)

    def __getitem__(self, subscript):
        return self.neurons[subscript]


class Loss(Layer):
    def __init__(self, loss_function=MSE):
        self.neurons = [None]
        self.loss_function = loss_function
        super().__init__(1)

    def append_to(self, prev_layer):
        self.prev_layer = prev_layer
        self.neurons[0] = partial(self.loss_function, prediction=prev_layer.neurons[0])

    def dense_append_to(self, prev_layer):
        assert prev_layer.num_neurons == 1
        return self.append_to(prev_layer)

    def diff(self, wrt, at, label):
        if isinstance(self.neurons[0], partial):
            self.neurons[0] = self.neurons[0](label=label)
        return self.neurons[0].diff(wrt, at)

    def eval(self, at, label):
        if isinstance(self.neurons[0], partial):
            self.neurons[0] = self.neurons[0](label=label)
        return self.neurons[0].eval(at)

    def train(self, inputs, label, learning_rate=0.01, threshold=0.01):
        if isinstance(self.neurons[0], partial):
            self.neurons[0] = self.neurons[0](label=label)
        print(f"Training with a weight change threshold of {threshold*100}%")

        training = True
        dist = 1e10
        while training:
            orig_weights = self.all_layer_weights
            grad = self.layer_grad({**inputs, **orig_weights})[0]
            new_weights = {
                key: val - learning_rate * grad[key]
                for key, val in orig_weights.items()
            }
            self.set_all_layer_weights(new_weights)
            new_dist = manhattan(new_weights, orig_weights)
            percentage_dist_change = (dist - new_dist) / dist
            prediction = self.prev_layer.neurons[0].eval({**inputs, **new_weights})
            loss = self.neurons[0].eval({**inputs, **new_weights})
            print(
                f"\r Prediction: {prediction}|Loss: {loss}| % weight change : {percentage_dist_change}"
            )

            training = False if percentage_dist_change < threshold else True
            dist = new_dist