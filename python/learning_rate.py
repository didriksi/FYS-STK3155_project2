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
            self.parameters['b'] = base
            decay = kwargs['decay']
            self.parameters['d'] = decay
            self.function = lambda step: base/(1 + step * decay)

        self.name = self.function.__doc__ or ' '.join(
            [f"{key}:{f'{value:.2e}' if isinstance(value, float) and int(value) == value else value}"
                                                            for key, value in self.parameters.items()])

        if 'name' in kwargs:
            self.name = kwargs['name']

    def ramp_up(self, ramp_up_steps, ramp_up_type="linear"):
        self.base_function = self.function
        ramp_up_to = self.function(0)
        def function(step):
            if step < ramp_up_steps:
                return (step/ramp_up_steps) * (ramp_up_to - ramp_up_to/10) + ramp_up_to/10
            else:
                return self.base_function(step-ramp_up_steps)

        self.function = function
        self.name = f"/{self.name}"

        return self

    def plot(self, max_steps, filename="learning_rate"):
        learning_rates = []
        for step in np.arange(max_steps):
            learning_rates.append(self.function(step))

        plt.plot(learning_rates)
        plt.xlabel("Steps")
        plt.ylabel("$\\gamma$")
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

if __name__ == '__main__':
    learning_rate = Learning_rate(base=1e-3, decay=1/2000).ramp_up(200)
    learning_rate.plot(10000)



