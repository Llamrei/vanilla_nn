from vanilla_nn.layers import Layer
from vanilla_nn.functions import Var

Input = Layer(2)
Input.inputs([Var("x"), Var("y")])
Hidden = Layer(5)
Hidden.dense_append_to(Input)
Output = Layer(1)
Output.dense_append_to(Hidden)
print(Output[0].diff("x", {"x": 7, "y": 10, **Output.all_layer_weights}))
Output.set_all_layer_weights({key: 0 for key in Output.all_layer_weights})
print(Output[0].diff("x", {"x": 7, "y": 10, **Output.all_layer_weights}))

print("Debug")