import csv
import sys
import numpy as np

from models import DataSet
from predict import Predictor
from rich.console import Console


console = Console()
prefix = '[purple][bold][Accuracy][/][/]'


class Accuracy:
    def __init__(self, theta_file_path: str, data_set: DataSet):
        """ Initialize the Accuracy class """
        self.theta_file_path = theta_file_path
        self.data_set = data_set
        self.theta0, self.theta1 = None, None

    def load_thetas(self):
        """ Load thetas from the file """
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

    def compute_accuracy(self):
        """ Compute the accuracy of the model """
        if self.theta0 is None or self.theta1 is None:
            raise ValueError('Thetas are not loaded')

        predictor = Predictor()
        price_values = np.array([row[1] for row in self.data_set.values])

        sum_real_price = price_values.sum()
        sum_pred_price = 0

        km_values = [row[0] for row in self.data_set.values]
        for km in km_values:
            sum_pred_price += int(predictor.estimate_price(self.theta0, self.theta1, km))

        console.log(
            f'{prefix} Sum of real prices: [bold]{sum_real_price}[/]\n'
            f'{prefix} Sum of predicted prices: [bold]{sum_pred_price}[/]\n'
        )

        accuracy_percentage = 100 - abs(((sum_pred_price - sum_real_price) / sum_real_price) * 100)
        return accuracy_percentage


def main():
    """ Main function """
    def ask_for_data_file_path():
        """ Ask the user for the path to the data file """
        while True:
            input_file = console.input(
                'Enter the path to the [gray][bold]file containing data[/][/]: '
            )
            try:
                with open(input_file) as _:
                    break
            except FileNotFoundError:
                console.print(
                    '[red][bold]Error:[/] [gray]File not found[/]'
                )
                continue
            except Exception as e:
                console.print(
                    f'[red][bold]Error:[/] [gray]Unexpected error - [/][white]{e}[/]'
                )
                sys.exit(1)
        return input_file

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

    def ask_for_theta_file_path():
        """ Ask the user for the path to the file containing thetas """
        while True:
            theta_file_path = console.input(
                'Enter the path to the [gray][bold]file containing thetas[/][/]: '
            )
            try:
                with open(theta_file_path) as _:
                    pass
            except FileNotFoundError:
                console.print(
                    '[red][bold]Error:[/] [gray]File not found[/]'
                )
                continue
            except Exception as e:
                console.print(
                    f'[red][bold]Error:[/] [gray]Unexpected error - [/][white]{e}[/]'
                )
                sys.exit(1)
            break
        return theta_file_path

    def check_thetas(accuracy: Accuracy):
        """ Check if the thetas are as expected """
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

    data_file_path = ask_for_data_file_path()
    theta_file_path = ask_for_theta_file_path()

    data_set = DataSet(data_file_path)
    check_data(data_set)

    accuracy = Accuracy(theta_file_path, data_set)
    check_thetas(accuracy)

    accuracy_percentage = accuracy.compute_accuracy()
    console.print(
        f'{prefix} The accuracy of the model is [green]{str(accuracy_percentage)}%[/]'
    )


if __name__ == '__main__':
    main()
