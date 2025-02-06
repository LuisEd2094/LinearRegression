# train.py
import csv, math
import sys


def load_data(filename):
    mileage = []
    price = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            mileage.append(float(row[0]))
            price.append(float(row[1]))
    return mileage, price


def evaluate_model(mileage, price, theta0, theta1):
    predicted_prices = [theta0 + (theta1 * m) for m in mileage]
    # MSE (Mean Squared Error)= standard deviation = difference of values from mean = spread around the mean
    # RMSE (Root Mean Squared Error)= standard error = difference in difference between values = error of prediction
    m = len(mileage)
    mse = sum((price[i] - predicted_prices[i]) ** 2 for i in range(m)) / m
    rmse = math.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    return mse, rmse, predicted_prices

def load_thetas():
    try:
        with open("thetas.txt", "r") as file:
            theta0, theta1 = map(float, file.read().split(","))
    except FileNotFoundError:
        theta0, theta1 = 0, 0
    return theta0, theta1

# Main function
if __name__ == "__main__":
    try:
        file_name = sys.argv[1] if len(sys.argv) > 1  else "data.csv"
        print(f"Evaluating model using data from {file_name}")
        theta0, theta1 = load_thetas()
        mileage, price, = load_data(file_name)
        evaluate_model(mileage, price, theta0, theta1)
    except Exception as e:
        print(f"An error occurred: {e}")
