from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

class Plot(ABC):
    @abstractmethod
    def plot(self, x, y):
        #define the function plot and pass, the others functions will implement it 
        pass

class PlotDecorator(Plot):
    #Herency of class Plot
    def __init__(self, a_plot):
        self._next_plot = a_plot

    def plot(self, x, y):
        self._next_plot.plot(x, y)

class PlotBasic(Plot):
    def plot(self, x, y):
        plt.figure()
        plt.plot(x, y)

class PlotTitle(PlotDecorator):
    #function for the title of the graphic
    def __init__(self, title, a_plot):
        self.title = title
        super(PlotTitle, self).__init__(a_plot)
    def plot(self, x, y):
        super().plot(x, y) # make plot() of next plot first
        # or self._next_plot.plot(x,y)
        plt.title(self.title)

class PlotGrid(PlotDecorator):
    def plot(self, x, y):
        super().plot(x, y)
        plt.grid(True)

class PlotLegend(PlotDecorator):
    #the function that adds the legend
    def plot(self, x, y):
        super().plot(x, y)
        plt.legend()

class PlotLabels(PlotDecorator):
    #Add the labels to the graphic, to the x and the y 
    def __init__(self, label_x, label_y, a_plot):
        self.label_x = label_x
        self.label_y = label_y
        super().__init__(a_plot)

    def plot(self, x, y):
        super().plot(x, y)
        plt.xlabel(self.label_x)
        plt.ylabel(self.label_y)

class PlotSize(PlotDecorator):
    #function for define the size of the graphic, the width and the height using cm
    def __init__(self, width_cm, height_cm, a_plot):
        self.width = width_cm / 2.54
        self.height = height_cm / 2.54
        super().__init__(a_plot)

    def plot(self, x, y):
        super().plot(x, y)
        plt.gcf().set_size_inches(self.width, self.height)

class PlotMean(PlotDecorator):
    def plot(self, x, y):
        super().plot(x, y)
        plt.plot([x[0], x[-1]], [y.mean(), y.mean()], 'b:')

class PlotGlobalExtrema(PlotDecorator):
    def plot(self, x, y):
        super().plot(x, y)
        plt.plot(x[np.argmin(y)], y.min(), 'bo', label='global min')
        plt.plot(x[np.argmax(y)], y.max(), 'ro', label='global max')
        plt.text(x[np.argmin(y)] + 0.5, y.min(), str(round(y.min(), 2)))
        plt.text(x[np.argmax(y)] + 0.5, y.max(), str(round(y.max(), 2)))

class PlotLocalExtrema(PlotDecorator):
    def plot(self, x, y):
        super().plot(x, y)
        idx_maxima = np.where(np.logical_and(y[1:-1]-y[:-2] > 0, y[1:-1]-y[2:] > 0))[0] + 1
        idx_minima = np.where(np.logical_and(y[1:-1]-y[:-2] < 0, y[1:-1]-y[2:] < 0))[0] + 1
        plt.plot(x[idx_maxima], y[idx_maxima], 'gx', label='local max')
        plt.plot(x[idx_minima], y[idx_minima], 'kx', label='local min')


#Execution test
if __name__ == '__main__':
    x = np.linspace(-10*np.pi, 10*np.pi, 1000)
    y = np.sin(x)/x

    p_basic = PlotBasic()
    p_grid = PlotGrid(PlotTitle('basic + grid + title', PlotBasic()))
    p_extrema = PlotTitle('basic + global + local extrema + title',
                PlotGlobalExtrema(
                    PlotLocalExtrema(
                    PlotBasic())))
    p_full = PlotSize(24, 20,
            PlotTitle('full',
                PlotLabels('x', 'sin(x)/x',
                PlotLegend(
                    PlotGrid(
                    PlotMean(
                        PlotGlobalExtrema(
                        PlotLocalExtrema(
                            PlotBasic()))))))))

    for p in [p_basic, p_grid, p_extrema, p_full]:
        p.plot(x, y)
        plt.show()

    #declaration of variables for the graphics
    x1 = np.linspace(-10*np.pi, 10*np.pi, 1000)
    y1 = np.sin(x1)/x1
    x2 = np.linspace(0, 2*np.pi, 500)
    y2 = np.sin(x2)**2
    x3 = np.linspace(-np.pi, np.pi, 200)
    y3 = np.cos(x3) * np.sin(x3)

    #There are some enviroments that dont need the plt.show() but we put it just in case
    p_extrema.plot(x1,y1)
    plt.show()
    p_full.plot(x2,y2)
    plt.show()
    p_extrema.plot(x3,y3)
    plt.show()
