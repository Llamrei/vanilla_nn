from vanilla_nn.layers import Layer
from vanilla_nn.layers import Loss
from vanilla_nn.functions import Var
from vanilla_nn.functions import ReLU
from vanilla_nn.functions import PReLU
from vanilla_nn.metrics import manhattan

from functools import partial
import logging


print("Building network")
Input = Layer(1)
Input.inputs([Var("x"),])
Hidden = Layer(5)
Hidden.dense_append_to(Input)
# Hidden_two = Layer(3)
# Hidden_two.dense_append_to(Hidden,ReLU)
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

# Fit y = -10 + 5*x
y = [-10+5*(i['x']) for i in x]

# Fit y = x^3
# y = [i['x']*i['x']*i['x'] for i in x]

# Fit 15 to inputs (5,10)
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.animation import FFMpegWriter


class TwoDPlotter:
#TODO: Track real time between plots and maybe also loss?
    def __init__(self, x, y, saving=False):
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

        self.saving = False
        if saving:
            self.writer = FFMpegWriter(fps=1)
            self.saving = True

    def __call__(self, new_x, new_y):
        new_x = [i['x'] for i in new_x]
        offsets = list(zip(new_x, new_y))
        self.points.set_offsets(offsets)
        self.update_blit()

        if self.saving:
            self.writer.grab_frame()

    def update_blit(self):
        self.fig.canvas.restore_region(self.cache)
        self.real_ax.draw_artist(self.points)
        self.fig.canvas.blit(self.fig.bbox)

    def save(self, filepath):
        if self.saving:
            return self.writer.saving(self.fig, filepath, 300)
        else:
            raise ValueError("Cannot save a plotter not configured to save")

        

plotter = TwoDPlotter(x,y, saving=True)
with plotter.save('nn_anim.mp4'):
    #TODO: Think about a training algorithm that can tell if loss went up (over a rolling threshold?) -> took too large a step in gradient direction
    Output.train(x, y, plotter=plotter, learning_rate=0.001, open_run=True)

# plotter = TwoDPlotter(x,y)
# Output.train(x, y, plotter=plotter, learning_rate=0.001, open_run=True)