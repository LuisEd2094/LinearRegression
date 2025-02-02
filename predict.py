# predict.py
NORMALIZED_MILEAGE = 10000

# Load the trained thetas
def load_thetas():
    with open("thetas.txt", "r") as file:
        theta0, theta1 = map(float, file.read().split(","))
    return theta0, theta1

# Predict price based on user input
def predict_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

# Main function
if __name__ == "__main__":
    theta0, theta1 = load_thetas()
    mileage = float(input("Enter mileage: ")) / NORMALIZED_MILEAGE
    estimated_price = predict_price(mileage, theta0, theta1)
    print(f"Estimated Price: {estimated_price}")
