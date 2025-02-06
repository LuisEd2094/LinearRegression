# train.py
import csv, math
import matplotlib.pyplot as plt
import sys

def plot_data(mileage, price, predicted_prices):
    plt.figure(figsize=(10, 6))

    # Scatter plot for actual data
    plt.scatter(mileage, price, color="blue", label="Actual Data")

    # Plot the predicted line (regression line)
    plt.plot(mileage, predicted_prices, color="red", label="Prediction Line")

    plt.xlabel("Mileage (normalized)")
    plt.ylabel("Price (normalized)")
    plt.title("Actual vs Predicted Prices")

    plt.legend()

    # Show the plot
    plt.savefig("price_vs_mileage_plot.png")


def plot_history(theta0_history, theta1_history):
    plt.figure(figsize=(10, 6))

    # Plot theta0
    plt.subplot(2, 1, 1)  # Create a 2x1 grid of plots, first subplot
    plt.plot(range(len(theta0_history)), theta0_history, color="blue")
    plt.title("Evolution of theta0 over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("theta0")

    # Plot theta1
    plt.subplot(2, 1, 2)  # Second subplot
    plt.plot(range(len(theta1_history)), theta1_history, color="red")
    plt.title("Evolution of theta1 over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("theta1")

    plt.tight_layout()  # Adjust the layout to prevent overlap
    plt.savefig("theta_history_plot.png")


def load_data(filename):
    mileage = []
    price = []
    max_mileage = float('-inf')
    min_mileage = float('inf')
    max_price = float('-inf')
    min_price = float('inf')
    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            # We need some sort of normalization to make sure the mileage values are not too large
            # We can divide by 10000 to make the values smaller
            mileage.append(float(row[0]))
            price.append(float(row[1]))
            max_mileage = max(max_mileage, mileage[-1])
            min_mileage = min(min_mileage, mileage[-1])
            max_price = max(max_price, price[-1])
            min_price = min(min_price, price[-1])
    range_mileage = max_mileage - min_mileage
    range_price = max_price - min_price
    mileage = [(m - min_mileage) / range_mileage for m in mileage]
    price = [(p - min_price) / range_price for p in price]

    return mileage, price, min_mileage, range_mileage, min_price, range_price


# Theta0 means the y-intercept of the line, the price at that point
# Theta1 means the slope, how much the price changes for each KM
# Since the more mileage a car has, the less it costs, the slope should be negative
# Hypothesis function
def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


# Gradient Descent Algorithm
def train_model(mileage, price, learning_rate=0.01, iterations=5000):
    # we need to find the best values for theta0 and theta1
    # we can do this by minimizing the cost function
    # Start with random values set to 0, making no predictions
    # Over enough iterations, values will tend towards the best values

    theta0, theta1 = 0, 0
    m = len(mileage)
    theta0_history = []
    theta1_history = []
    for _ in range(iterations):

        sum_errors0 = sum(
            estimate_price(mileage[i], theta0, theta1) - price[i] for i in range(m)
        )
        sum_errors1 = sum(
            (estimate_price(mileage[i], theta0, theta1) - price[i]) * mileage[i]
            for i in range(m)
        )

        # Update the values of theta0 and theta1 using gradient descent
        # each iteration will bring us closer to the best values
        theta0 -= learning_rate * (1 / m) * sum_errors0
        theta1 -= learning_rate * (1 / m) * sum_errors1
        theta0_history.append(theta0)
        theta1_history.append(theta1)

    return theta0, theta1, theta0_history, theta1_history


def get_predicted_prices(mileage, theta0, theta1):
    return [theta0 + (theta1 * m) for m in mileage]


def save_thetas(theta0, theta1, min_mileage, range_mileage, min_price, range_price):
    theta1_real = theta1 * (range_price / range_mileage)
    
    # Denormalize theta0
    theta0_real = min_price + (theta0 * range_price) - (theta1_real * min_mileage)

    with open("thetas.txt", "w") as file:
        file.write(f"{theta0_real},{theta1_real}")

# Main function
if __name__ == "__main__":
    try:
        file_name = sys.argv[1] if len(sys.argv) > 1  else "data.csv"
        print(f"Training model using data from {file_name}")
        mileage, price, min_mileage, range_mileage, min_price, range_price = load_data(file_name)
        theta0, theta1, theta0_history, theta1_history = train_model(mileage=mileage, price=price, iterations=10000)
        predicted_prices = get_predicted_prices(mileage, theta0, theta1)
        save_thetas(theta0, theta1, min_mileage, range_mileage, min_price, range_price)
        print(f"Training complete! theta0 = {theta0}, theta1 = {theta1}")
        plot_data(mileage, price, predicted_prices)
        plot_history(theta0_history, theta1_history)
    except Exception as e:
        print(f"An error occurred: {e}")
