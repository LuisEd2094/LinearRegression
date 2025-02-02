# train.py
import csv, math
import matplotlib.pyplot as plt
NORMALIZED_MILEAGE = 10000
# Theta0 means the y-intercept of the line, the price at that point
# Theta1 means the slope, how much the price changes for each KM
# Since the more mileage a car has, the less it costs, the slope should be negative

# Load dataset
def load_data(filename):
    mileage = []
    price = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            # We need some sort of normalization to make sure the mileage values are not too large
            # We can divide by 10000 to make the values smaller
            mileage.append(float(row[0]) / NORMALIZED_MILEAGE)
            price.append(float(row[1]))
    return mileage, price

# Hypothesis function
def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

# Gradient Descent Algorithm
def train_model(mileage, price, learning_rate=0.01, iterations=10000):
    # we need to find the best values for theta0 and theta1
    # we can do this by minimizing the cost function
    # Start with random values set to 0, making no predictions
    # Over enough iterations, values will tend towards the best values

    theta0, theta1 = 0, 0
    m = len(mileage)

    for _ in range(iterations):

        sum_errors0 = sum(estimate_price(mileage[i], theta0, theta1) - price[i] for i in range(m))
        sum_errors1 = sum((estimate_price(mileage[i], theta0, theta1) - price[i]) * mileage[i] for i in range(m))

        # Update the values of theta0 and theta1 using gradient descent
        # each iteration will bring us closer to the best values
        theta0 -= learning_rate * (1/m) * sum_errors0
        theta1 -= learning_rate * (1/m) * sum_errors1

    return theta0, theta1

def evaluate_model(mileage, price, theta0, theta1):
    predicted_prices = [theta0 + (theta1 * m) for m in mileage]
    #MSE = standard deviation = difference of values from mean = spread around the mean
    #RMSE = standard error = difference in difference between values = error of prediction
    m = len(mileage)
    # Calculate Mean Squared Error (MSE)

    mse = sum((price[i]- predicted_prices[i]) ** 2 for i in range(m)) / m
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = math.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    return mse, rmse, predicted_prices

# Save thetas
def save_thetas(theta0, theta1):
    with open("thetas.txt", "w") as file:
        file.write(f"{theta0},{theta1}")


def plot_data(mileage, price, predicted_prices):
    plt.figure(figsize=(10, 6))
    
    # Scatter plot for actual data
    plt.scatter(mileage, price, color='blue', label='Actual Data')
    
    # Plot the predicted line (regression line)
    plt.plot(mileage, predicted_prices, color='red', label='Prediction Line')
    
    # Labels and title
    plt.xlabel('Mileage 10s thousand(normalized)')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices')

    # Show legend
    plt.legend()

    # Show the plot
    plt.savefig("price_vs_mileage_plot.png")

# Main function
if __name__ == "__main__":
    try:
        mileage, price = load_data("data.csv")
        theta0, theta1 = train_model(mileage, price)
        mse, rmse, predicted_prices = evaluate_model(mileage, price, theta0, theta1)
        save_thetas(theta0, theta1)
        print(f"Training complete! theta0 = {theta0}, theta1 = {theta1}")
        plot_data(mileage, price, predicted_prices)
    except Exception as e:
        print(f"An error occurred: {e}")