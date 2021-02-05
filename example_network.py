from vanilla_nn.layers import Layer
from vanilla_nn.layers import Loss
from vanilla_nn.functions import Var
from vanilla_nn.metrics import manhattan

import logging

Input = Layer(1)
Input.inputs([Var("x"),])
Hidden = Layer(5)
Hidden.dense_append_to(Input)
# Hidden_two = Layer(3)
# Hidden_two.dense_append_to(Hidden)
Condense = Layer(1)
Condense.dense_append_to(Hidden)
Output = Loss()
Output.append_to(Condense)

# Fit y = x^2
num_train_el = 20
x = [{"x":i - num_train_el/2} for i in range(num_train_el+1)]
y = [2+5*i['x'] for i in x]

# Fit 15 to inputs (5,10)
import matplotlib.pyplot as plt
import matplotlib.animation as anim

class TwoDPlotter:
    def __init__(self, x, y):
        plt.ion()
        self.fig = plt.figure()
        self.real_ax = self.fig.subplots()
        new_x = [i['x'] for i in x]
        self.real_ax.scatter(new_x,y, c='red', label='Training Data')
        self.points = self.real_ax.scatter([],[], c='green', label='Neural Net')
        self.fig.legend()
        plt.show()
        plt.draw()


    def __call__(self, new_x, new_y):
        new_x = [i['x'] for i in new_x]
        offsets = list(zip(new_x, new_y))
        self.points.set_offsets(offsets)
        self.fig.canvas.draw()

        

plotter = TwoDPlotter(x,y)
Output.train(x, y, plotter=plotter, learning_rate=0.001, open_run=True)
print(
Output.prev_layer[0].eval({"x":25, **Output.all_layer_weights})
)