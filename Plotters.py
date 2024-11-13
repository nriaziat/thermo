import matplotlib.pyplot as plt
from matplotlib import rcParams
import do_mpc

class AdaptiveParameterPlotter:
    def __init__(self, adaptive_model, plot_indices: list[bool| int]):
        self._adaptive_model = adaptive_model
        self._plot_indices = plot_indices
        assert len(plot_indices) == len(adaptive_model.labels)
        self.n = sum([1 for i in plot_indices if i])
        if self.n == 1:
            self.fig, self.axs = plt.subplots(1, figsize=(16, 9), sharex=True)
            self.lines = []
            line, = self.axs.plot([], [], label=adaptive_model.labels[0])
            self.lines.append(line)
            self.axs.set_ylabel(adaptive_model.labels[0])
            self.axs.legend()
        else:
            self.fig, self.axs = plt.subplots(self.n, figsize=(16, 9), sharex=True)
            self.lines = []
            self.cis = []
            self._labels = [label for i, label in enumerate(adaptive_model.labels) if plot_indices[i]]
            for i, (ax, label) in enumerate(zip(self.axs, self._labels)):
                line, = self.axs[i].plot([], [], label=label)
                self.lines.append(line)
                self.axs[i].set_ylabel(label)
                self.axs[i].legend()
        self.fills = []

    def plot(self):
        for f in self.fills:
            f.remove()
        self.fills = []
        if self.n == 1:
            y = [x[1] for x in self._adaptive_model.data[self._adaptive_model.labels[0]]]
            self.lines[0].set_data(range(len(self._adaptive_model.data[self._adaptive_model.labels[0]])), y)
            self.axs.relim()
            self.axs.autoscale_view()
        else:
            for i, label in enumerate(self._labels):
                y = [x[1] for x in self._adaptive_model.data[label]]
                self.lines[i].set_data(range(len(self._adaptive_model.data[label])), y)
                self.axs[i].relim()
                self.axs[i].autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class GenericPlotter:
    def __init__(self, n_plots: int, labels: list[str], x_label: str, y_labels: list[str]):
        plt.ion()
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'
        rcParams['axes.grid'] = True
        rcParams['lines.linewidth'] = 2.0
        rcParams['axes.labelsize'] = 'xx-large'
        rcParams['xtick.labelsize'] = 'xx-large'
        rcParams['ytick.labelsize'] = 'xx-large'
        self._data = {labels[i]: [] for i in range(n_plots)}
        assert len(labels) == n_plots == len(y_labels)
        self.fig, self.axs = plt.subplots(n_plots, figsize=(16, 9), sharex=True)
        self.lines = []
        for i, ax in enumerate(self.axs):
            line, = ax.plot([], [], label=labels[i])
            self.lines.append(line)
            ax.set_ylabel(y_labels[i])
            ax.legend()
        self.axs[-1].set_xlabel(x_label)

    def plot(self, y: list[float], defl_hist=None):
        for i, label in enumerate(self._data.keys()):
            if label == 'Deflection' and defl_hist is not None:
                self._data[label] = defl_hist
            else:
                self._data[label].append(y[i])
            self.lines[i].set_data(range(len(self._data[label])), self._data[label])
            self.axs[i].relim()
            self.axs[i].autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



class MPCPlotter:
    def __init__(self, mpc_data: do_mpc.data.Data, isotherm_temps: list[float]):
        plt.ion()
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'
        rcParams['axes.grid'] = True
        rcParams['lines.linewidth'] = 2.0
        rcParams['axes.labelsize'] = 'xx-large'
        rcParams['xtick.labelsize'] = 'xx-large'
        rcParams['ytick.labelsize'] = 'xx-large'
        self.fig, self.axs = plt.subplots(3, sharex=True, figsize=(16, 9))
        for i in range(1, len(self.axs)-1):
            self.axs[i].sharex(self.axs[0])
            self.axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        self.line_plots = []
        self.graphics = do_mpc.graphics.Graphics(mpc_data)
        isotherm_colors = ['lightcoral', 'orangered', 'orange']
        for i, (temp, color) in enumerate(zip(isotherm_temps, isotherm_colors)):
            self.graphics.add_line(var_type='_x', var_name=f'width_{i}', axis=self.axs[0], label=f'{temp:.2f}C', color=isotherm_colors[i%len(isotherm_colors)])
        self.graphics.add_line(var_type='_x', var_name='tip_lead_dist', axis=self.axs[0], color='g', label='Tip Lead Distance')
        self.graphics.add_line(var_type='_tvp', var_name='defl_meas', axis=self.axs[1], color='purple', label='Measured')
        self.graphics.add_line(var_type='_z', var_name='deflection', axis=self.axs[1], color='b', linestyle='--', label='Predicted')
        self.graphics.add_line(var_type='_u', var_name='u', axis=self.axs[2], color='b')
        # self.graphics.add_line(var_type='_tvp', var_name='d', axis=self.axs[3])

        self.axs[0].set_ylabel(r'$w~[\si[per-mode=fraction]{\milli\meter}]$')
        self.axs[0].legend()
        self.axs[1].set_ylabel(r'$d~[\si[per-mode=fraction]{\milli\meter}]$')
        self.axs[1].legend()
        self.axs[2].set_ylabel(r"$u~[\si[per-mode=fraction]{\milli\meter\per\second}]$")
        self.axs[-1].set_xlabel(r'$t~[\si[per-mode=fraction]{\second}]$')
        self.fig.align_ylabels()


    def plot(self, t_ind=None):
        if t_ind is None:
            self.graphics.plot_results()
            self.graphics.plot_predictions()
            self.graphics.reset_axes()
            self.axs[-1].set_ylim(0, 10)

        else:
            self.graphics.plot_results(t_ind)
            self.graphics.plot_predictions(t_ind)
            self.graphics.reset_axes()

    def __del__(self):
        plt.ioff()
