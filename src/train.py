import sys
import csv
import numpy as np

from models import DataSet
from predict import Predictor


class LinearRegression:
	def __init__(self, data_set: DataSet, learning_rate: float, iterations: int):
		""" Initialize the LinearRegression class """
		self.data_set = data_set
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.X = [row[0] for row in data_set.values]
		self.y = [row[1] for row in data_set.values]
		self.normalized_X = self.normalize(self.X)
		self.normalized_y = self.normalize(self.y)
		self.theta0, self.theta1 = 0, 0

	def normalize(self, array: list):
		""" Normalize the data """
		return (array - np.mean(array)) / np.std(array)

	def denormalize(self, normalized_theta0: float, normalized_theta1: float):
		""" Denormalize the theta values """
		theta1 = normalized_theta1 * (np.std(self.y) / np.std(self.X))
		theta0 = np.mean(self.y) - theta1 * np.mean(self.X) + normalized_theta0 * np.std(self.y)
		return theta0, theta1

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

		for i in range(self.iterations):
			normalized_y_pred = predictor.estimate_price(normalized_theta0, normalized_theta1, self.normalized_X)

			normalized_gradient_theta0 = (1 / m) * np.sum(normalized_y_pred - self.normalized_y)
			normalized_gradient_theta1 = (1 / m) * np.sum(self.normalized_X * (normalized_y_pred - self.normalized_y))

			normalized_theta0 -= self.learning_rate * normalized_gradient_theta0
			normalized_theta1 -= self.learning_rate * normalized_gradient_theta1

			if i % 100 == 0:
				cost = self.compute_cost(normalized_theta0, normalized_theta1)
				print(f'Train: Iteration #{i} has a cost of {cost}')

		self.theta0, self.theta1 = self.denormalize(normalized_theta0, normalized_theta1)

	def export_thetas(self):
		""" Export the theta values to a file """
		while True:
			output_file = input('Enter the path to the file to save thetas: ')
			try:
				with open(output_file, 'w') as _:
					break
			except Exception as e:
				print(f'Error: Unexpected error - {e}')
				continue

		try:
			with open(output_file, 'w') as file:
				writer = csv.DictWriter(file, fieldnames=['theta0', 'theta1'])
				writer.writeheader()
				writer.writerow({'theta0': self.theta0, 'theta1': self.theta1})
		except Exception as e:
			sys.exit(f'Error: Unexpected error - {e}')

		print(f'Train: Theta values saved to {output_file}')


def main():
	""" Main function """
	def ask_for_data_file_path():
		""" Ask the user for the path to the data file """
		while True:
			input_file = input('Enter the path to the data file: ')
			try:
				with open(input_file) as _:
					break
			except FileNotFoundError:
				print('Error: File not found')
				continue
			except Exception as e:
				sys.exit(f'Error: Unexpected error - {e}')
		return input_file

	def check_data(data_set: DataSet):
		""" Check if the data is as expected """
		try:
			print('DataSet: Checking data...')
			data_set.validate_data()
			print('DataSet: Data validation successful')
		except AssertionError as e:
			sys.exit(f'Error: Data validation failed - {e}')
		except Exception as e:
			sys.exit(f'Error: Unexpected error - {e}')

	input_file = ask_for_data_file_path()
	data_set = DataSet(input_file)

	check_data(data_set)

	linear_regression = LinearRegression(data_set, 0.01, 1000)
	linear_regression.train()
	linear_regression.export_thetas()


if __name__ == '__main__':
	main()
