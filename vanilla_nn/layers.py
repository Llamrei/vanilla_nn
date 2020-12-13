from .functions import Sum
from .functions import Prod
from .functions import Const
from .functions import Var

from functools import reduce
import logging
import sys

from random import random

logging.basicConfig(filename="debug.log", encoding="utf-8", level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logger = logging.getLogger(__name__)


class Layer:
    # Has a list of neurons which are all just Functions
    def __init__(self, num_neurons, weight_init_func=random):
        self.neurons = [(None)] * num_neurons
        self.num_neurons = num_neurons
        self.nodes_namespace = None
        # Weights are defined looking back
        self.weights = {}
        self.prev_layer = None
        self.weight_init_func = weight_init_func

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
        return weights

    def set_all_layer_weights(self, weights):
        prev_layer = self
        while prev_layer:
            for key in prev_layer.weights:
                prev_layer.weights[key] = weights[key]
            prev_layer = prev_layer.prev_layer

    def __iter__(self):
        return iter(self.neurons)

    def __getitem__(self, subscript):
        return self.neurons[subscript]