import csv
import sys
import argparse
import numpy as np

from models import DataSet
from predict import Predictor
from rich.console import Console


console = Console()
prefix = '[purple][bold][Accuracy][/][/]'


class Accuracy:
    """
    Accuracy class to compute the accuracy of the model
    Calculates the R-squared value of the model
    
    Attributes:
        - theta_file_path (str): Path to the file containing the theta values
        - data_set (DataSet): DataSet object containing the data
    """
    
    def __init__(self, theta_file_path: str, data_set: DataSet):
        """ Initialize the Accuracy class """
        self.theta_file_path = theta_file_path
        self.data_set = data_set
        self.theta0, self.theta1 = None, None

    def load_thetas(self):
        """
        Load the theta values from the file
        
        Raises:
            - ValueError: If the theta file path is not set
            - ValueError: If the theta file has incorrect columns
            - ValueError: If the theta file has incorrect number of rows
            - ValueError: If theta0 is not a float
            - ValueError: If theta1 is not a float
        """
        if not self.theta_file_path:
            raise ValueError('Theta file path is not set')

        theta_file = open(self.theta_file_path)
        reader = csv.DictReader(theta_file)
        columns = reader.fieldnames

        if columns != ['theta0', 'theta1']:
            raise ValueError('Theta file has incorrect columns')

        theta_values = [row for row in reader]
        if len(theta_values) != 1:
            raise ValueError('Theta file has incorrect number of rows')

        for row in theta_values:
            try:
                float(row['theta0'])
            except ValueError:
                raise ValueError('theta0 is not a float')
            try:
                float(row['theta1'])
            except ValueError:
                raise ValueError('theta1 is not a float')

            self.theta0 = float(row['theta0'])
            self.theta1 = float(row['theta1'])

    def compute_rsquared(self):
        """
        Compute the R-squared value of the model
        It explains the amount of variation between the actual and predicted values
        The closer the value is to 1, the better the model
        
        Returns:
            - rsquared (float): R-squared value of the model
        """
        if self.theta0 is None or self.theta1 is None:
            raise ValueError('Thetas are not loaded')

        predictor = Predictor()
        X = [row[0] for row in self.data_set.values]
        y = [row[1] for row in self.data_set.values]
        square_sum = 0
        square_residual_sum = 0

        for i in range(len(X)):
            y_pred = predictor.estimate_price(self.theta0, self.theta1, X[i])
            square_sum += (y[i] - np.mean(y)) ** 2
            square_residual_sum += (y[i] - y_pred) ** 2

        rsquared = 1 - (square_residual_sum / square_sum)
        return rsquared


def main():
    """ Main function """
    parser = argparse.ArgumentParser(description='Compute the accuracy of the model')
    parser.add_argument('data_file', type=str, help='Path to the file containing the data')
    parser.add_argument('theta_file', type=str, help='Path to the file containing theta values')
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

    def check_thetas(accuracy: Accuracy):
        """
        Check if the thetas are valid
        
        Raises:
            - ValueError: If the thetas are not loaded
            - Exception: If an unexpected error occurs
        """
        try:
            console.log(
                f'{prefix} [white]Loading thetas...[/]'
            )
            accuracy.load_thetas()
            console.log(
                f'{prefix} [white]Thetas loaded [green]successfully[/][/]'
            )
        except ValueError as e:
            console.print(
                f'[red][bold]Error:[/] [gray]Thetas loading failed - [/][white]{e}[/]'
            )
            sys.exit(1)
        except Exception as e:
            console.print(
                f'[red][bold]Error:[/] [gray]Unexpected error - [/][white]{e}[/]'
            )
            sys.exit(1)

    data_file_path = args.data_file
    theta_file_path = args.theta_file

    data_set = DataSet(data_file_path)
    check_data(data_set)

    accuracy = Accuracy(theta_file_path, data_set)
    check_thetas(accuracy)

    rsquared = accuracy.compute_rsquared()
    percentage = rsquared * 100
    percentage_color = 'green' if percentage > 60 else 'red'
    console.print(
        f'{prefix} The accuracy of the model is [{percentage_color}]{percentage:.2f}%[/]'
    )


if __name__ == '__main__':
    main()
