import csv

class Predictor:
	def __init__(self, data):
		self.data = data

	def validate_data(self):
		columns = self.data.fieldnames

		# Check if the column names are as expected
		assert columns == ['km', 'price'], 'Column names are not as expected'

		# Check if the data is as expected, float values, no missing values
		for row in data:
			assert row['km'].replace('.', '', 1).isdigit(), 'km is not a float'
			assert row['price'].replace('.', '', 1).isdigit(), 'price is not a float'

if __name__ == '__main__':
	input_file = 'data/data.csv'
	data = csv.DictReader(open(input_file))
	predictor = Predictor(data)

	try:
		predictor.validate_data()
	except AssertionError as e:
		print('Data is not valid:', e)
	except Exception as e:
		print('An error occurred:', e)
