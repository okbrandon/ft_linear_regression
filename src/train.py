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
    LinearRegression class to train the model
    
    Attributes:
        - parameters (Parameters): Parameters object containing the data set, learning rate, and number of iterations
    """
    
    class Parameters:
        """
        Parameters class to store the parameters for the model
        
        Attributes:
            - data_set (DataSet): DataSet object containing the data
            - learning_rate (float): Learning rate for the model (default: 0.01)
            - iterations (int): Number of iterations for the model (default: 10000)
            - live_plotting (bool): Enable live plotting (default: False)
        """
        def __init__(self, data_set: DataSet, learning_rate: float, iterations: int = 1000, live_plotting: bool = False):
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
        Function used to normalize the data
        Normalizing the data helps in faster convergence
        It scales the data between 0 and 1
        
        Args:
            - array (Iterable): List of values to be normalized
        """
        array = np.array(array)
        min_value = array.min()
        max_value = array.max()
        return (array - min_value) / (max_value - min_value)

    def denormalize(self, normalized_theta0, normalized_theta1):
        """
        Function used to denormalize the theta values
        Denormalizing the theta values helps in getting the actual values
        
        Args:
            - normalized_theta0 (float): Normalized value of theta0
            - normalized_theta1 (float): Normalized value of theta1
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
        Function used to compute the RMSE value
        RMSE is the Root Mean Squared Error
        It calculates how close the predicted values are to the actual values
        
        Args:
            - theta0 (float): Value of theta0
            - theta1 (float): Value of theta1
        """
        predictor = Predictor()
        y_pred = [predictor.estimate_price(theta0, theta1, x) for x in self.normalized_X]
        
        mse = np.square(np.subtract(self.normalized_y, y_pred)).mean()
        rmse = np.sqrt(mse)
        
        return rmse

    def train(self):
        """
        Function used to train the model
        It uses the Gradient Descent algorithm to minimize the cost function
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
        Function used to export the theta values to a file
        
        Raises:
            - ValueError: If the output file path is not set
            - Exception: If there is an unexpected error while writing to the file
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
        """
        Function used to plot the data
        It plots the data points and the regression line
        """
        predictor = Predictor()

        if not os.path.exists('./img'):
            os.makedirs('./img')

        plt.scatter(self.X, self.y, color='purple', label='Data points', marker='*')
        plt.xlabel('Mileage (in km)')
        plt.ylabel('Price')
        plt.title('Mileage vs Price - Linear Regression')
        plt.legend()
        plt.savefig('./img/data_plot.png')

        # Plot the data with regression line
        plt.plot(self.X, [predictor.estimate_price(self.theta0, self.theta1, x) for x in self.X], color='pink', label='Regression line')
        plt.legend()
        plt.savefig('./img/data_plot_with_regression.png')
        
        # Plot the normalized cost history
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
        """
        Function used to plot the data live
        
        Args:
            - axis (Axes): Axis object for the plot
            - thetas (tuple): Tuple containing the theta values
            - iteration (int): Current iteration number
        """
        predictor = Predictor()
        
        axis[0].cla()
        axis[0].scatter(self.X, self.y, color='purple', label='Data points', marker='*')
        axis[0].plot(self.X, [predictor.estimate_price(thetas[0], thetas[1], x) for x in self.X], color='pink', label='Regression line')
        axis[0].set_xlabel('Mileage (in km)')
        axis[0].set_ylabel('Price')
        axis[0].set_title('Mileage vs Price - Linear Regression')
        
        axis[1].plot(range(iteration), self.cost_history, color='purple', label='Cost history')
        axis[1].set_xlabel('Iterations')
        axis[1].set_ylabel('RMSE')
        axis[1].set_title('Iterations vs RMSE')


def main():
    """ Main function """
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('data_file', type=str, help='Path to the file containing the data')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01, help='Learning rate for the model')
    parser.add_argument('-i', '--iterations', type=int, default=10000, help='Number of iterations for the model')
    parser.add_argument('-lp', '--live_plot', action='store_true', help='Enable live plotting')
    args = parser.parse_args()

    def check_data(data_set: DataSet):
        """
        Check if the data is valid
        
        Raises:
            - AssertionError: If the data is not valid
            - Exception: If an unexpected error occurs
        """
        try:
            console.log(
                f'{prefix} Validating the data...'
            )
            data_set.validate_data()
            console.log(
                f'{prefix} Data validated [green]successfully[/]'
            )
        except AssertionError as e:
            console.print(
                f'[red][bold]Error:[/] [gray]Data verification went bad - [/][white]{e}[/]'
            )
            sys.exit(1)
        except Exception as e:
            console.print(
                f'[red][bold]Error:[/] [gray]Unexpected error - [/][white]{e}[/]'
            )
            sys.exit(1)

    input_file = args.data_file
    learning_rate = args.learning_rate
    iterations = args.iterations
    live_plot = args.live_plot
    data_set = DataSet(input_file)

    check_data(data_set)

    parameters = LinearRegression.Parameters(data_set, learning_rate, iterations, live_plot)
    linear_regression = LinearRegression(parameters)
    linear_regression.train()
    linear_regression.export_thetas()
    linear_regression.plot()


if __name__ == '__main__':
    main()
