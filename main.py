import csv
import sys

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
