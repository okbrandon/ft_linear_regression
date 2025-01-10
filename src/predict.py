import csv
import sys
import argparse

from rich.console import Console


console = Console()
prefix = '[purple][bold][Predictor][/][/]'


class Predictor:
    """
    Predictor class to predict the price of a car based on its mileage
    
    Attributes:
        - theta_file_path (str): Path to the file containing the theta values
        - theta0 (float): Value of theta0
        - theta1 (float): Value of theta1
    """
    
    def __init__(self, theta_file_path: str = ""):
        """ Initialize the Predictor class """
        self.theta_file_path = theta_file_path
        self.theta0, self.theta1 = 0, 0

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

    def ask_mileage(self):
        """
        Function used to ask the user for the mileage
        
        Returns:
            - mileage (float): Mileage of the car
        """
        while True:
            mileage = console.input(
                'Enter the mileage [gray][bold](in km)[/][/]: '
            )
            try:
                mileage = float(mileage)
                if mileage < 0:
                    raise ValueError('Mileage cannot be negative')
                break
            except ValueError as e:
                console.print(
                    f'[red][bold]Error:[/] [gray]Invalid mileage - [/][white]{e}[/]'
                )
        return mileage

    def estimate_price(self, theta0, theta1, mileage):
        """
        Estimate the price of the car based on the mileage
        The formula used is: price = theta0 + (theta1 * mileage)
        
        Args:
            - theta0 (float): Value of theta0
            - theta1 (float): Value of theta1
            - mileage (float): Mileage of the car
        """
        return float(theta0) + (float(theta1) * mileage)


def main():
    """ Main function """
    parser = argparse.ArgumentParser(description='Predict the price of a car based on its mileage')
    parser.add_argument('-f', '--theta_file', type=str, help='Path to the file containing theta values')
    args = parser.parse_args()

    predictor = Predictor(args.theta_file)

    if args.theta_file:
        try:
            console.log(
                f'{prefix} [white]Loading thetas...[/]'
            )
            predictor.load_thetas()
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

    mileage = predictor.ask_mileage()
    price = predictor.estimate_price(predictor.theta0, predictor.theta1, mileage)

    console.print(
        f'{prefix} [white]Estimated price for [bold]{mileage} km[/] is [bold]${price}[/][/]'
    )


if __name__ == '__main__':
    main()
