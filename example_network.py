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
# Fit 15 to inputs (5,10)
Output.train({"x": 5, "y": 10}, 15)