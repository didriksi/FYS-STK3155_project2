import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

class Learning_rate:
    def __init__(self, **kwargs):
        self.parameters = {}
        if 'learning_rate' in kwargs:
            if callable(kwargs['learning_rate']):
                self.function = kwargs['learning_rate']
                if self.function.__doc__ is None:
                    self.parameters['(0)'] = self.function(0)
            else:
                self.function = lambda step: kwargs['learning_rate']
                self.parameters['flat'] = kwargs['learning_rate']
        elif 'decay' in kwargs:
            if 'base' in kwargs:
                base = kwargs['base']
            else:
                base = 0.01
            self.parameters['base'] = base
            decay = kwargs['decay']
            self.parameters['decay'] = decay
            self.function = lambda step: base/(1 + step * decay)

        self.name = self.function.__doc__ or ' '.join([f"{key}:{f'{value:.2e}' if isinstance(value, float) else value}" for key, value in self.parameters.items()])

        if 'name' in kwargs:
            self.name = kwargs['name']

        self.base_function = self.function

    def ramp_up(self, ramp_up_steps, ramp_up_type="linear"):
        ramp_up_to = self.function(0)
        if ramp_up_type == 'linear':
            def function(step):
                if step < ramp_up_steps:
                    return (step+1/ramp_up_steps+1) * ramp_up_to
                else:
                    return self.base_function(step-ramp_up_steps)
        else:
            raise NotImplementedError("Haven't implemented any other ramp up types than linear")

        self.function = function
        self.name = f"ramp->{self.name}"

        return self

    def plot(self, max_steps, filename="../plots/learning_rate"):
        learning_rates = []
        steps = np.arange(max_steps)
        for step in steps:
            learning_rates.append(self.function(step))

        plt.plot(steps, learning_rates)
        plt.title(self.name)
        plt.savefig(f"../plots/{filename}.png")

    def compile(self, max_steps):
        learning_rates = []
        for step in np.arange(max_steps):
            learning_rates.append(self.function(step))
        self.learning_rates = np.array(learning_rates)
        self.function = lambda step: self.learning_rates[step]
        self.function.__doc__ = self.name
        return self.function