
### `predictor.py`
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset (size, bedrooms, price)
data = pd.DataFrame({
    "size": [850, 900, 1000, 1200, 1500, 1800, 2000],
    "bedrooms": [2, 2, 3, 3, 4, 4, 5],
    "price": [200000, 210000, 250000, 280000, 350000, 400000, 450000]
})

X = data[["size", "bedrooms"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

def predict_price(size, bedrooms):
    return model.predict([[size, bedrooms]])[0]

if __name__ == "__main__":
    size = int(input("Enter house size (sqft): "))
    bedrooms = int(input("Enter number of bedrooms: "))
    price = predict_price(size, bedrooms)
    print(f"🏠 Estimated Price: {price:.2f} AED")
