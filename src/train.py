import sys
import csv
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import DataSet
from predict import Predictor
from rich.console import Console


console = Console()
prefix = '[purple][bold][Train][/][/]'


class LinearRegression:
    def __init__(self, data_set: DataSet, learning_rate: float, iterations: int = 1000):
        """ Initialize the LinearRegression class """
        self.data_set = data_set
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.X = [row[0] for row in data_set.values]
        self.y = [row[1] for row in data_set.values]
        self.normalized_X = self.normalize(self.X)
        self.normalized_y = self.normalize(self.y)
        self.theta0, self.theta1 = 0, 0

    def normalize(self, array):
        """ Normalize the data """
        array = np.array(array)
        min_value = array.min()
        max_value = array.max()
        return (array - min_value) / (max_value - min_value)

    def denormalize(self, normalized_theta0, normalized_theta1):
        """ Denormalize the theta values """
        min_price_value = min(self.y)
        max_price_value = max(self.y)
        min_km_value = min(self.X)
        max_km_value = max(self.X)

        denormalized_theta0 = normalized_theta0 * (max_price_value - min_price_value) + min_price_value + (normalized_theta1 * min_km_value * (min_price_value - max_price_value) / (max_km_value - min_km_value))
        denormalized_theta1 = (normalized_theta1 * (max_price_value - min_price_value) / (max_km_value - min_km_value))

        return denormalized_theta0, denormalized_theta1

    def compute_cost(self, theta0, theta1):
        """ Compute the cost """
        m = len(self.normalized_X)
        predictions = theta0 + theta1 * self.normalized_X
        cost = (1 / (2 * m)) * np.sum((predictions - self.normalized_y) ** 2)
        return cost

    def train(self):
        """ Train the model """
        normalized_theta0, normalized_theta1 = 0, 0
        m = len(self.normalized_X)
        predictor = Predictor()

        previous_values = {
            'theta0': 0,
            'theta1': 0
        }

        for i in range(self.iterations):
            previous_values['theta0'], previous_values['theta1'] = normalized_theta0, normalized_theta1

            normalized_y_pred = predictor.estimate_price(normalized_theta0, normalized_theta1, self.normalized_X)

            normalized_gradient_theta0 = (1 / m) * np.sum(normalized_y_pred - self.normalized_y)
            normalized_gradient_theta1 = (1 / m) * np.sum(self.normalized_X * (normalized_y_pred - self.normalized_y))

            normalized_theta0 -= self.learning_rate * normalized_gradient_theta0
            normalized_theta1 -= self.learning_rate * normalized_gradient_theta1

            if previous_values['theta0'] == normalized_theta0 and previous_values['theta1'] == normalized_theta1:
                console.log(
                    f'{prefix} Converged after [bold]{i}[/] iterations'
                )
                break

            if i % 100 == 0:
                cost = self.compute_cost(normalized_theta0, normalized_theta1)
                console.log(
                    f'{prefix} Iteration #[bold]{i}[/] has a cost of [bold]{cost}[/]'
                )

        self.theta0, self.theta1 = self.denormalize(normalized_theta0, normalized_theta1)

    def export_thetas(self):
        """ Export the theta values to a file """
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

        """ Plot the data """
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

        console.log(
            f'{prefix} Plots saved to [bold]img[/]'
        )


def main():
    """ Main function """
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('data_file', type=str, help='Path to the file containing the data')
    args = parser.parse_args()

    def check_data(data_set: DataSet):
        """ Check if the data is as expected """
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
    data_set = DataSet(input_file)

    check_data(data_set)

    linear_regression = LinearRegression(data_set, 0.1, 10000)
    linear_regression.train()
    linear_regression.export_thetas()
    linear_regression.plot()


if __name__ == '__main__':
    main()
