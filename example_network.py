from vanilla_nn.layers import Layer
from vanilla_nn.layers import Loss
from vanilla_nn.functions import Var
from vanilla_nn.metrics import manhattan

import logging

Input = Layer(2)
Input.inputs([Var("x"), Var("y")])
Hidden = Layer(5)
Hidden.dense_append_to(Input)
Condense = Layer(1)
Condense.dense_append_to(Hidden)
Output = Loss()
Output.append_to(Condense)
weights = Output.all_layer_weights
Output.diff("w2_7", {"x": 7, "y": 10, **weights}, 15)

Output.train({"x": 5, "y": 10}, 15)