import pandas as pd
import numpy as np

# Number of users and products
num_users = 100
num_products = 50

# Generate random data for the dataset
data = {
    'userId': np.random.randint(1, num_users + 1, size=1000),  # 1000 interactions
    'productId': np.random.randint(1, num_products + 1, size=1000),
    'ratings': np.random.randint(1, 6, size=1000),  # Ratings from 1 to 5
    'timestamp': pd.to_datetime(np.random.randint(1620000000, 1670000000, size=1000), unit='s')  # Random timestamps between two dates
}

# Create a DataFrame from the generated data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('personalised_product_recommendation.csv', index=False)

print('Generated dataset saved to personalised_product_recommendation.csv')