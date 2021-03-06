from .functions import Identity, LReLU, PReLU, Sum
from .functions import Prod
from .functions import Exp
from .functions import Const
from .functions import Var
from .functions import Neg
from .functions import MSE
from .functions import ReLU
from .functions import Identity
from .metrics import manhattan

from functools import reduce
from functools import partial
import logging
import sys

from random import random, randrange

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler("debug.log"))

positive_rand = partial(randrange,start=0,stop=10)

class Layer:
    # Has a list of neurons which are all just Functions
    def __init__(self, num_neurons, weight_init_func=random, bias_init_func=random):
        self.neurons = [(None)] * num_neurons
        self.num_neurons = num_neurons
        self.nodes_namespace = None
        # Weights are defined looking back
        self.weights = dict()
        self.prev_layer = None
        self.weight_init_func = weight_init_func
        self.bias_init_func = bias_init_func

    def __str__(self):
        # Careful that this code is seperate from usage in the dense_append_to
        return f"{[(x, i+self.nodes_namespace[0]) for i, x in enumerate(self.neurons)]}"

    # TODO: Implement passing activation function
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
                trans = f"t{prev_name}_{neuron_name}"
                bias = f"b{neuron_name}"
                param = f"p{prev_name}_{neuron_name}"
                # TODO: Look into replacing linear weighting with alternative function?
                # weighted_previous_layer.append(Exp(prev_neuron, Var(weight)))
                weighted_previous_layer.append(Prod(ReLU( Sum(prev_neuron, Var(trans) )), Var(weight)))
                self.weights[weight] = self.weight_init_func()
                self.weights[bias] = self.bias_init_func()
                self.weights[trans] = self.bias_init_func()
                self.weights[param] = self.weight_init_func()
            self.neurons[neuron_idx] = Sum(reduce(Sum, weighted_previous_layer), Prod(Var(bias),Const(10)))

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
                    grad[neuron_idx][weight] = self.neurons[neuron_idx].diff(weight, at)
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
        self.neuron_generator = partial(self.loss_function, prediction=prev_layer.neurons[0])

    def dense_append_to(self, prev_layer):
        assert prev_layer.num_neurons == 1
        return self.append_to(prev_layer)

    def train(
        self,
        inputs_iter,
        labels_iter,
        plotter=None,
        open_run=False,
        learning_rate=0.001,
        perc_threshold=0.01,
        abs_threshold=1e-3,
    ):
        if open_run:
            print("Training indefinitely until KeyboardInterrupt")
        else:
            print(
                f"Training with a weight change threshold of {perc_threshold:}% | {abs_threshold} abs"
            )

        training = True
        loss = 1e10
        dist = 1e10
        num_inputs = len(inputs_iter)
        grads = [None] * num_inputs
        
        if plotter:
            predictions = [None] * num_inputs
            for idx, inputs in enumerate(inputs_iter):
                predictions[idx] = self.prev_layer.neurons[0].eval({**inputs, **self.all_layer_weights})
            plotter(inputs_iter,predictions)

        while training or open_run:
            orig_weights = self.all_layer_weights
            for idx, inputs in enumerate(inputs_iter):
                self.neurons[0] = self.neuron_generator(label=labels_iter[idx])
                grads[idx] = self.layer_grad({**inputs, **orig_weights})[0]
            grad = {
                key: sum([g[key] for g in grads])/num_inputs 
                for key in orig_weights
            }
            new_weights = {
                key: val - learning_rate * grad[key]
                for key, val in orig_weights.items()
            }
            self.set_all_layer_weights(new_weights)
            new_losses = [None] * num_inputs
            predictions = [None] * num_inputs
            for idx, inputs in enumerate(inputs_iter):
                predictions[idx] = self.prev_layer.neurons[0].eval({**inputs, **new_weights})
                self.neurons[0] = self.neuron_generator(label=labels_iter[idx])
                new_losses[idx] = self.neurons[0].eval({**inputs, **new_weights})
            new_dist = manhattan(orig_weights, new_weights)
            new_loss = sum(l for l in new_losses)/num_inputs
            
            percentage_loss_change = (new_loss - loss) / loss * 100
            percentage_dist_change = abs((new_dist - dist) / dist) * 100

            if plotter:
                plotter(inputs_iter,predictions)
            
            #TODO: Printing/keeping track of timestamps
            print(
                f"Average Loss: {new_loss} ({percentage_loss_change}%) | weight_movement: {percentage_dist_change}%"
            )

            training = (
                False
                if abs(percentage_loss_change) < perc_threshold or loss < new_loss
                else True
            )
            loss = new_loss
            dist = new_dist
        print("Finished training")