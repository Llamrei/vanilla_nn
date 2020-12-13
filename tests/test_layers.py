# Input = Layer(2)
# Input.inputs([Var("x"), Var("y")])
# Hidden = Layer(5)
# Hidden.dense_append_to(Input)
# Output = Layer(1)
# Output.dense_append_to(Hidden)
# weights = Output.all_layer_weights
# deriv = Output[0].diff("x", {"x": 7, "y": 10, **weights})