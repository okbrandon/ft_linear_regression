import csv
import sys


class Predictor:
	def __init__(self, theta_file_path: str = None):
		""" Initialize the Predictor class """
		self.theta_file_path = theta_file_path
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

	def ask_mileage(self):
		""" Ask the user for the mileage """
		if self.theta0 is None or self.theta1 is None:
			raise ValueError('Thetas are not loaded')

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


def main():
	""" Main function """
	def ask_for_theta_file_path():
		""" Ask the user for the path to the file containing thetas """
		while True:
			theta_file_path = input('Enter the path to the file containing thetas: ')
			try:
				with open(theta_file_path) as _:
					pass
			except FileNotFoundError:
				print('Error: File not found')
				continue
			except Exception as e:
				sys.exit(f'Error: Unexpected error - {e}')
			break
		return theta_file_path

	theta_file_path = ask_for_theta_file_path()
	predictor = Predictor(theta_file_path)

	try:
		print('Predictor: Loading thetas...')
		predictor.load_thetas()
		print('Predictor: Thetas loaded successfully')
	except ValueError as e:
		sys.exit(f'Error: Thetas loading failed - {e}')
	except Exception as e:
		sys.exit(f'Error: Unexpected error - {e}')

	mileage = predictor.ask_mileage()
	price = predictor.estimate_price(predictor.theta0, predictor.theta1, mileage)

	print(f'Predictor: Estimated price for {mileage} km is ${price}')


if __name__ == '__main__':
	main()
