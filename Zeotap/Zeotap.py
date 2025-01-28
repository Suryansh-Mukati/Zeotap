import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

customers = pd.read_csv('D:\coding\Zeotap\Customers.csv')
products = pd.read_csv('D:\coding\Zeotap\Products.csv')
transactions = pd.read_csv('D:\coding\Zeotap\Transactions.csv')

data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')

# ====================================
# Task 1: Exploratory Data Analysis (EDA)
# ====================================
print("Dataset Information:")
print(data.info())
print("\nDescriptive Statistics:")
print(data.describe())

plt.figure(figsize=(10, 6))
sns.countplot(x='Region', data=customers)
plt.title('Customer Distribution by Region')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='TotalValue', data=data, estimator=sum)
plt.title('Total Sales by Product Category')
plt.xticks(rotation=45)
plt.show()

print("\nSuggested Business Insights:")
print("1. Customers from certain regions contribute more to sales.")
print("2. High sales are observed in specific product categories.")
print("3. Seasonal trends may exist based on transaction dates.")
print("4. Top customers account for a significant portion of sales.")
print("5. Average transaction value varies significantly across product categories.")

# ====================================
# Task 2: Lookalike Model
# ====================================
encoded_customers = pd.get_dummies(customers.drop(columns=['CustomerName', 'SignupDate']))
scaler = StandardScaler()
scaled_customers = scaler.fit_transform(encoded_customers)

knn = NearestNeighbors(n_neighbors=4)
knn.fit(scaled_customers)

lookalikes = {}
for idx, customer_id in enumerate(customers['CustomerID'][:20]):
    distances, indices = knn.kneighbors([scaled_customers[idx]])
    recommendations = [(customers['CustomerID'][i], distances[0][j]) for j, i in enumerate(indices[0]) if i != idx]
    lookalikes[customer_id] = recommendations[:3]

lookalike_df = pd.DataFrame.from_dict(lookalikes, orient='index', columns=['Lookalike_1', 'Lookalike_2', 'Lookalike_3'])
lookalike_df.to_csv('FirstName_LastName_Lookalike.csv')
print("\nLookalike Model Results saved as 'FirstName_LastName_Lookalike.csv'")

# ====================================
# Task 3: Customer Segmentation / Clustering
# ====================================
clustering_data = data.groupby('CustomerID').agg({'TotalValue': 'sum', 'Quantity': 'sum'}).reset_index()
clustering_features = clustering_data[['TotalValue', 'Quantity']]

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(clustering_features)

db_index = davies_bouldin_score(clustering_features, labels)
print(f"\nDavies-Bouldin Index: {db_index}")

plt.figure(figsize=(10, 6))
plt.scatter(clustering_features['TotalValue'], clustering_features['Quantity'], c=labels, cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Total Transaction Value')
plt.ylabel('Quantity Purchased')
plt.show()

print("\nCustomer Segmentation and Clustering complete.")
