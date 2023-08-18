import numpy as np
import pandas as pd
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
# from sklearn.externals import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import gzip



# Define column names for the data
columns = ['userId', 'productId', 'ratings', 'timestamp']

# Read the CSV file and store the data in a DataFrame
df = pd.read_csv('all_csv_files.csv', names=columns, nrows=50_000)

# Print descriptive statistics of the DataFrame
print(df.describe())

# Print the first few rows of the DataFrame
print(df.head())

# Drop the 'timestamp' column from the DataFrame
df.drop('timestamp', axis=1, inplace=True)

# Print information about the DataFrame
print(df.info())

# Calculate and print descriptive statistics of the 'ratings' column
print(df['ratings'].describe().transpose())

# Print the minimum rating
print(f"Minimum rating is: {df['ratings'].min()}")

# Print the maximum rating
print(f"Maximum rating is: {df['ratings'].max()}")

# Check for missing values across columns
print('Number of missing values across columns:\n', df.isnull().sum())

# Generate a histogram of the 'ratings' column
plt.hist(df["ratings"])
# plt.show()

# Calculate the number of ratings per user and sort them in descending order
most_rated = df.groupby('userId').size().sort_values(ascending=False)[:10]

# Print the top 10 users based on ratings
print('Top 10 users based on ratings: \n', most_rated)

# Count the number of ratings for each user
counts = df.userId.value_counts()

# Filter the DataFrame to include only users who have rated 15 or more items
df_final = df[df.userId.isin(counts[counts >= 15].index)]

# Print the number of users who have rated 25 or more items
print('Number of users who have rated 25 or more items =', len(df_final))

# Print the number of unique users in the final data
print('Number of unique users in the final data = ', df_final['userId'].nunique())

# Print the number of unique products in the final data
print('Number of unique products in the final data = ', df_final['productId'].nunique())

# Remove duplicate user entries to ensure unique user IDs in the final DataFrame
df_final = df_final.drop_duplicates(subset='userId')

# Create a pivot table with 'userId' as the index, 'productId' as the columns, and 'ratings' as the values
# Fill missing values with 0
final_ratings_matrix = df_final.pivot(index='userId', columns='productId', values='ratings').fillna(0)

# Print the first few rows of the final ratings matrix
print(final_ratings_matrix.head())

# Print the shape of the final ratings matrix
print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)

# Calculate the number of non-zero ratings in the final ratings matrix
given_num_of_ratings = np.count_nonzero(final_ratings_matrix)

# Calculate the possible number of ratings in the final ratings matrix
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]

# Calculate the density of the ratings matrix (ratio of given ratings to possible ratings)
density = (given_num_of_ratings / possible_num_of_ratings)
density *= 100

# Print the density of the ratings matrix as a percentage
print('density: {:4.2f}%'.format(density))
print(given_num_of_ratings)
print(possible_num_of_ratings)

# Splitting the dataset into train and test data with a test size of 20% and a random state of 0
train_data, test_data = train_test_split(df_final, test_size=0.2, random_state=0)

# Grouping the train data by 'productId' and aggregating the count of 'userId', then resetting the index
train_data = train_data.groupby('productId').agg({'userId': 'count'}).reset_index()

# Renaming the 'userId' column to 'score' in the train_data DataFrame
train_data.rename(columns={'userId': 'score'}, inplace=True)

# Printing the first 40 rows of the train_data DataFrame
print(train_data.head(40))

# Sorting the train_data DataFrame by 'score' and 'productId' in descending and ascending order respectively
train_data_sort = train_data.sort_values(['score', 'productId'], ascending=[0, 1])

# Generating a rank for each product based on the 'score' in descending order
train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first')

# Getting the top 5 recommendations from the sorted train_data DataFrame
popularity_recommendations = train_data_sort.head(5)

# Printing the top 5 recommendations
print(popularity_recommendations)
#### recommend base on popularity
# Defining a function to recommend products to a user
def recommend(user_id):
    # Using the popularity_recommendations DataFrame as the user recommendations
    user_recommendations = popularity_recommendations

    # Adding a 'userId' column with the specified user_id for which recommendations are being generated
    user_recommendations['userId'] = user_id

    # Reordering the columns, bringing 'userId' to the front
    cols = user_recommendations.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    user_recommendations = user_recommendations[cols]

    return user_recommendations


# List of user choices for recommendations
find_recom = [25, 140, 190]

# Iterating over each user ID in the find_recom list
for i in find_recom:
    print("The list of recommendations for the userId: %d\n" % i)
    print(recommend(i))
    print("\n")

# Combining the train_data and test_data DataFrames into df_CF and resetting the index
df_CF = pd.concat([train_data, test_data]).reset_index()

# Creating a pivot table from df_CF, with 'userId' as the index, 'productId' as columns, and 'ratings' as values, filling NaN values with 0
pivot_df = df_CF.pivot(index='userId', columns='productId', values='ratings').fillna(0)

# Adding a new 'user_index' column with sequential numbers as the index
pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)

# Setting 'user_index' as the new index of pivot_df
pivot_df.set_index(['user_index'], inplace=True)
pivot_matrix = pivot_df.to_numpy()

# Applying Singular Value Decomposition (SVD) on pivot_df, obtaining U, sigma, and Vt matrices
U, sigma, Vt = svds(pivot_matrix, k=10)

# Converting sigma into a diagonal matrix
sigma = np.diag(sigma)
print('Diagonal matrix: \n', sigma)

# Calculating the predicted ratings by performing matrix multiplication using U, sigma, and Vt
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Converting the predicted ratings into a DataFrame with the same columns as pivot_df
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=pivot_df.columns)
preds_df.head()

# Defining a function to recommend items to a user based on user ID, pivot_df, preds_df, and the number of recommendations
def recommend_items(userID, pivot_df, preds_df, num_recommendations):
    user_idx = userID - 1  # Index starts at 0
    sorted_user_ratings = pivot_df.iloc[user_idx].sort_values(ascending=False)
    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    temp.index.name = 'Recommended Items'
    temp.columns = ['user_ratings', 'user_predictions']
    temp = temp.loc[temp.user_ratings == 0]
    temp = temp.sort_values('user_predictions', ascending=False)
    print('\nBelow are the recommended items for user (user_id = {}):\n'.format(userID))
    print(temp.head(num_recommendations))


# Recommending items for a specific user (user ID = 4) with a specified number of recommendations
userID = 4
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations)

final_ratings_matrix.head()

final_ratings_matrix.mean().head()

preds_df.head()

preds_df.mean().head()

# Combining the average actual ratings and average predicted ratings into a DataFrame
rmse_df = pd.concat([final_ratings_matrix.mean(), preds_df.mean()], axis=1)
rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']
rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)
rmse_df.head()

# Calculating the Root Mean Square Error (RMSE) between the average actual ratings and average predicted ratings
RMSE = round((((rmse_df.Avg_actual_ratings - rmse_df.Avg_predicted_ratings) ** 2).mean() ** 0.5), 5)
print('\nRMSE SVD Model = {} \n'.format(RMSE))

# Recommending items for a specific user (user ID = 9) with a specified number of recommendations
userID = 9
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations)
