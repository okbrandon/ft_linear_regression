import sys
import csv
import os
import argparse
from typing_extensions import Iterable
import numpy as np
import matplotlib.pyplot as plt

from models import DataSet
from predict import Predictor
from rich.console import Console


console = Console()
prefix = '[purple][bold][Train][/][/]'


class LinearRegression:
    """
    Linear Regression class to train the model
    
    Attributes:
        parameters (Parameters): The parameters for the model
    """
    
    class Parameters:
        """
        Parameters class to store the parameters for the model
        
        Attributes:
            data_set (DataSet): The data set
            learning_rate (float): The learning rate (default: 0.01)
            iterations (int): The number of iterations (default: 10000)
            live_plotting (bool): Whether to enable live plotting
        """
        
        def __init__(self, data_set: DataSet, learning_rate: float, iterations: int = 1000, live_plotting: bool = False):
            """ Initialize the Parameters class """
            self.data_set = data_set
            self.learning_rate = learning_rate
            self.iterations = iterations
            self.live_plotting = live_plotting
    
    def __init__(self, parameters: Parameters):
        """ Initialize the LinearRegression class """
        self.parameters = parameters
        self.X = [row[0] for row in self.parameters.data_set.values]
        self.y = [row[1] for row in self.parameters.data_set.values]
        self.normalized_X = self.normalize(self.X)
        self.normalized_y = self.normalize(self.y)
        self.theta0, self.theta1 = 0, 0
        self.cost_history = []

    def normalize(self, array):
        """
        Function used to normalize the array.
        Normalization is useful to scale the data between 0 and 1.
        
        Args:
            array (Iterable): The array to normalize
        """
        array = np.array(array)
        min_value = array.min()
        max_value = array.max()
        return (array - min_value) / (max_value - min_value)

    def denormalize(self, normalized_theta0, normalized_theta1):
        """
        Function used to denormalize the theta values.
        It is useful to get the actual theta values from the normalized ones.
        
        Args:
            normalized_theta0 (float): The normalized theta0 value
            normalized_theta1 (float): The normalized theta1 value
        """
        price_range = max(self.y) - min(self.y)
        km_range = max(self.X) - min(self.X)
        min_price = min(self.y)
        min_km = min(self.X)

        denormalized_theta1 = normalized_theta1 * (price_range / km_range)
        denormalized_theta0 = (normalized_theta0 * price_range + min_price + 
                               normalized_theta1 * min_km * (min_price - max(self.y)) / km_range)

        return denormalized_theta0, denormalized_theta1

    def compute_rmse(self, theta0, theta1):
        """
        Function used to compute the Root Mean Squared Error (RMSE).
        Evaluating how close the predicted values are to the actual values from a scale of 0 to 1.
        
        Args:
            theta0 (float): The theta0 value
            theta1 (float): The theta1 value
        """
        predictor = Predictor()
        y_pred = [predictor.estimate_price(theta0, theta1, x) for x in self.normalized_X]
        
        mse = np.square(np.subtract(self.normalized_y, y_pred)).mean()
        rmse = np.sqrt(mse)
        
        return rmse

    def train(self):
        """
        Function used to train the model.
        It uses the Gradient Descent algorithm to minimize the cost function.
        
        The cost function is the Root Mean Squared Error (RMSE).
        """
        normalized_theta0, normalized_theta1 = 0, 0
        m = len(self.normalized_X)
        predictor = Predictor()
        
        figure, axis = None, None
        if self.parameters.live_plotting:
            plt.ion()
            figure, axis = plt.subplots(2, figsize=(6, 10))
            figure.suptitle('Linear Regression Live Plotting')

        for i in range(self.parameters.iterations):
            normalized_y_pred = predictor.estimate_price(normalized_theta0, normalized_theta1, self.normalized_X)

            normalized_gradient_theta0 = (1 / m) * np.sum(normalized_y_pred - self.normalized_y)
            normalized_gradient_theta1 = (1 / m) * np.sum(self.normalized_X * (normalized_y_pred - self.normalized_y))

            normalized_theta0 -= self.parameters.learning_rate * normalized_gradient_theta0
            normalized_theta1 -= self.parameters.learning_rate * normalized_gradient_theta1

            rmse = self.compute_rmse(normalized_theta0, normalized_theta1)
            self.cost_history.append(rmse)

            if i % 100 == 0:
                console.log(
                    f'{prefix} Iteration #[bold]{i}[/] has a RMSE of [bold]{rmse}[/]'
                )
                
                if figure is not None and axis is not None and self.parameters.live_plotting:
                    self.live_plot(axis, self.denormalize(normalized_theta0, normalized_theta1), i + 1)
                    
                    figure.canvas.draw()
                    figure.canvas.flush_events()

        self.theta0, self.theta1 = self.denormalize(normalized_theta0, normalized_theta1)

    def export_thetas(self):
        """
        Function used to 
        """
        while True:
            output_file = console.input(
                'Please enter the path where you want [cyan][bold]the thetas[/] to be saved[/]: '
            )
            try:
                with open(output_file, 'w') as _:
                    break
            except Exception as e:
                console.print(
                    f'[red][bold]Error:[/] [gray]Unexpected error - [/][white]{e}[/]'
                )
                continue

        try:
            with open(output_file, 'w') as file:
                writer = csv.DictWriter(file, fieldnames=['theta0', 'theta1'])
                writer.writeheader()
                writer.writerow({'theta0': self.theta0, 'theta1': self.theta1})
        except Exception as e:
            console.print(
                    f'[red][bold]Error:[/] [gray]Unexpected error - [/][white]{e}[/]'
            )
            sys.exit(1)

        console.log(
            f'{prefix} Theta values saved to [bold]{output_file}[/]'
        )

    def plot(self):
        """ Plot the data """
        predictor = Predictor()

        if not os.path.exists('./img'):
            os.makedirs('./img')

        plt.scatter(self.X, self.y, color='purple', label='Data points', marker='*')
        plt.xlabel('Mileage (in km)')
        plt.ylabel('Price')
        plt.title('Mileage vs Price - Linear Regression')
        plt.legend()
        plt.savefig('./img/data_plot.png')

        """ Plot the data with regression line """
        plt.plot(self.X, [predictor.estimate_price(self.theta0, self.theta1, x) for x in self.X], color='pink', label='Regression line')
        plt.legend()
        plt.savefig('./img/data_plot_with_regression.png')
        
        """ Plot the normalized cost history """
        plt.clf()
        plt.plot(range(self.parameters.iterations), self.cost_history, color='purple', label='Cost history')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('Iterations vs RMSE')
        plt.legend()
        plt.savefig('./img/cost_history_plot.png')

        console.log(
            f'{prefix} Plots saved to directory [bold]img[/]'
        )
    
    def live_plot(self, axis, thetas, iteration: int):
        """ Live plot the data """
        predictor = Predictor()
        
        axis[0].cla()
        axis[0].scatter(self.X, self.y, color='purple', label='Data points', marker='*')
        axis[0].plot(self.X, [predictor.estimate_price(thetas[0], thetas[1], x) for x in self.X], color='pink', label='Regression line')
        axis[0].set_xlabel('Mileage (in km)')
        axis[0].set_ylabel('Price')
        axis[0].set_title('Mileage vs Price - Linear Regression')
        
        axis[1].plot(range(iteration),