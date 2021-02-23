from vanilla_nn.layers import Layer
from vanilla_nn.layers import Loss
from vanilla_nn.functions import Var
from vanilla_nn.metrics import manhattan

import logging


print("Building network")
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


num_train_el = 20
x = [{"x":i - num_train_el/2} for i in range(num_train_el+1)]

# Fit y = 5x
# y = [5*i['x'] for i in x]

# Fit y = 10 + 5x
# y = [10+5*i['x'] for i in x]

# Fit y = 10 + 5*(x-2)
y = [10+5*(i['x']-2) for i in x]

# Fit y = x^3
# y = [i['x']*i['x']*i['x'] for i in x]

# Fit 15 to inputs (5,10)
import matplotlib.pyplot as plt
import matplotlib.animation as anim

class TwoDPlotter:
#TODO: with blitting
    def __init__(self, x, y):
        plt.ion()
        self.fig = plt.figure()
        self.real_ax = self.fig.subplots()
        self.real_ax.grid('both')
        plt.show()
        new_x = [i['x'] for i in x]
        self.real_ax.scatter(new_x,y, c='red', label='Training Data')
        self.points = self.real_ax.scatter([],[], c='green', label='Neural Net')
        self.fig.legend()
        
        self.cache = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.fig.canvas.draw()


    def __call__(self, new_x, new_y):
        new_x = [i['x'] for i in new_x]
        offsets = list(zip(new_x, new_y))
        self.points.set_offsets(offsets)
        self.update_blit()

    def update_blit(self):
        self.fig.canvas.restore_region(self.cache)
        self.real_ax.draw_artist(self.points)
        self.fig.canvas.blit(self.fig.bbox)

        

plotter = TwoDPlotter(x,y)
Output.train(x, y, plotter=plotter, learning_rate=0.001, open_run=True)
print(
Output.prev_layer[0].eval({"x":25, **Output.all_layer_weights})
)