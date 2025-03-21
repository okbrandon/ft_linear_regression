<h1 align="center">📈 ft_linear_regression</h1>

<p align="center">
	<b><i>ft_linear_regression is a 42 School project designed to introduce the fundamentals of machine learning. The goal is to implement a simple linear regression algorithm using gradient descent to predict car prices based on their mileage.</i></b><br>
</p>

<p align="center">
	<img alt="Top used language" src="https://img.shields.io/github/languages/top/okbrandon/ft_linear_regression?color=success"/>
	<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/okbrandon/ft_linear_regression"/>
</p>

## 📚 Table of Contents

- [📚 Table of Contents](#-table-of-contents)
- [📣 Introduction](#-introduction)
- [📝 Usage](#-usage)
- [💁 Features](#-features)
- [📎 References](#-references)

## 📣 Introduction

**ft_linear_regression** is a project aimed at introducing the basics of machine learning through the implementation of a simple linear regression algorithm. The goal is to predict car prices based on their mileage using gradient descent.

- Understand the principles of linear regression.
- Implement gradient descent for optimizing the model.
- Analyze and visualize data to improve predictions.

This project is part of the 42 School curriculum post common core, focusing on foundational machine learning concepts and practical implementation.

## 📝 Usage

Here are the main commands for managing the project:

- **Create a virtual environment activate it**
  ```sh
  python3 -m virtualenv venv
  source venv/bin/activate
  ```

- **Install the requirements**
  ```sh
  pip3 install -r requirements.txt
  ```

- **Start the training process**
  ```sh
  python3 src/train.py ./data/data.csv
  ```

- **Run the predict program**
  ```sh
  python3 src/predict.py -f ./data/thetas.csv
  ```

- **Run the accuracy program**
  ```sh
  python3 src/accuracy.py ./data/data.csv ./data/thetas.csv
  ```

## 💁 Features

Our **ft_linear_regression** project includes the following features:

1. **Data Preprocessing**
   - Load and preprocess data from CSV files.
   - Handle missing values and normalize data for better performance.
2. **Linear Regression Model**
   - Implement a simple linear regression model from scratch.
   - Use gradient descent to optimize the model parameters.
3. **Training and Evaluation**
   - Train the model using the provided dataset.
   - Evaluate the model's performance using metrics like Mean Squared Error (MSE).
4. **Prediction**
   - Predict car prices based on mileage using the trained model.
   - Save and load model parameters for future predictions.
5. **Visualization**
   - Visualize the data and the regression line using matplotlib.
   - Plot training progress and error reduction over iterations.

## 📎 References

- [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
- [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)

and some more docs...

[⬆ Back to Top](#-table-of-contents)

## 🌏 Meta

**bsoubaig** – bsoubaig@student.42nice.fr
