import csv

class Predictor:
	def __init__(self, data):
		self.data = data

	def validate_data(self):
		reader = csv.DictReader(self.data)
		columns = reader.fieldnames

		# Check if the column names are as expected
		assert columns == ['km', 'price'], 'Column names are not as expected'

		# Check if the data is as expected, float values, no missing values
		for row in reader:
			assert row['km'].replace('.', '', 1).isdigit(), 'km is not a float'
			assert row['price'].replace('.', '', 1).isdigit(), 'price is not a float'

		self.data.seek(0)

if __name__ == '__main__':
	input_file = 'data/data.csv'
	predictor = Predictor(open(input_file))

	try:
		predictor.validate_data()

	except AssertionError as e:
		print('Data is not valid:', e)
	except Exception as e:
		print('An error occurred:', e)
