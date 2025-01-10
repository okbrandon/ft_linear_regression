import csv


class DataSet:
    """
    DataSet class to validate the data and store the values
    
    Attributes:
        - data_path (str): Path to the data file
        - data (file): File object of the data file
        - values (list): List to store the values
    """

    def __init__(self, data_path: str):
        """ Initialize the DataSet class """
        self.data_path = data_path
        self.data = open(data_path)
        self.values = []

    def validate_data(self):
        """
        Function used to validate the data
        
        Raises:
            - AssertionError: If the column names are not as expected
            - AssertionError: If the data is not as expected, float values, no missing values
        """
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
