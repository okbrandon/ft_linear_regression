import csv
import sys
import numpy as np

class DataSet:
	def __init__(self, data_path: str):
		self.data_path = data_path
		self.data = open(data_path)
		self.values = []

	def validate_data(self):
		reader = csv.DictReader(self.data)
		columns = reader.fieldnames

		# Check if the column names are as expected
		assert columns == ['km', 'price'], 'Column names are not as expected'

		# Check if the data is as expected, float values, no missing values
		for row in reader:
			assert row['km'].replace('.', '', 1).isdigit(), 'km is not a float'
			assert row['price'].replace('.', '', 1).isdigit(), 'price is not a float'

			self.values.append((float(row['km']), float(row['price'])))

		self.data.seek(0)

class Predictor:
	def __init__(self):
		pass

	def ask_mileage(self):
		while True:
			mileage = input('Enter mileage: ')
			try:
				mileage = float(mileage)
				if mileage < 0:
					raise ValueError('Mileage cannot be negative')
				break
			except ValueError as e:
				print(f'Error: Invalid mileage - {e}')
		return mileage

	def estimate_price(self, theta0: float, theta1: float, mileage: float):
		return theta0 + (theta1 * mileage)

class LinearRegression:
	def __init__(self, data_set: DataSet, learning_rate: float, iterations: int):
		self.data_set = data_set
		self.learning_rate = learning_rate
		self.iterations = iterations

		self.X = [row[0] for row in data_set.values]
		self.normalized_X = self.normalize(self.X)
		self.min_X = np.min(self.X)
		self.max_X = np.max(self.X)

		self.y = [row[1] for row in data_set.values]
		self.normalized_y = self.normalize(self.y)
		self.min_y = np.min(self.y)
		self.max_y = np.max(self.y)

	def normalize(self, array: list):
		return (array - np.min(array)) / (np.max(array) - np.min(array))

	def denormalize(self, normalized_theta0: float, normalized_theta1: float):
		theta1 = normalized_theta1 * (self.max_X - self.min_X) / (self.max_y - self.min_y)
		theta0 = normalized_theta0 * (self.max_y - self.min_y) - theta1 * self.min_X
		return theta0, theta1

	def train(self):
		normalized_theta0, normalized_theta1 = 0, 0
		m = len(self.normalized_X)
		predictor = Predictor()

		for _ in range(self.iterations):
			normalized_y_pred = predictor.estimate_price(normalized_theta0, normalized_theta1, self.normalized_X)
			normalized_theta0 -= self.learning_rate * (1 / m) * np.sum(normalized_y_pred - self.normalized_y)
			normalized_theta1 -= self.learning_rate * (1 / m) * np.sum((normalized_y_pred - self.normalized_y) * self.normalized_X)

		theta0, theta1 = self.denormalize(normalized_theta0, normalized_theta1)
		return theta0, theta1

if __name__ == '__main__':
	input_file = 'data/data.csv'
	theta0, theta1 = 0, 0

	data_set = DataSet(input_file)
	try:
		print('DataSet: Checking data...')
		data_set.validate_data()
		print('DataSet: Data validation successful')
	except AssertionError as e:
		sys.exit(f'Error: Data validation failed - {e}')
	except Exception as e:
		sys.exit(f'Error: Unexpected error - {e}')

	predictor = Predictor()
	mileage = predictor.ask_mileage()

	price = predictor.estimate_price(theta0, theta1, mileage)
	print(f'Predictor: Predicting price for mileage {mileage} - ${price}')

	linear_regression = LinearRegression(data_set, 0.15, 1000)
	theta0, theta1 = linear_regression.train()

	price = predictor.estimate_price(theta0, theta1, mileage)
	print(f'LinearRegression: Predicting price for mileage {mileage} - ${price}')
