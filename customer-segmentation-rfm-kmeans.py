#!/usr/bin/env python
# coding: utf-8

# ### Import the Relevant Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import warnings
import joblib
warnings.filterwarnings('ignore')


# 
# ### Import Libraries and Load Dataset

# In[2]:


# Load the dataset from a remote Excel file, parsing 'InvoiceDate' as datetime
df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx', parse_dates=['InvoiceDate'])

# Display the first few rows of the dataset
df.head()


# ## Data Exploration and Pre-Processing

# In[3]:


# Print the shape of the dataset
df.shape


# In[4]:


# Display concise summary information about the dataset
df.info()


# In[5]:


# Print the column names of the dataset
df.columns


# In[6]:


# Check the number of unique values in each column
df.nunique()


# In[8]:


print("There are {} duplicated values.".format(df.duplicated().sum()))
df[df.duplicated(keep=False)].head(4)


# In[9]:


new_data = df.drop_duplicates()
new_data


# In[10]:


new_data.duplicated().sum()


# In[11]:


# Check the data types of each column
data_types = new_data.dtypes
print(data_types)


# In[12]:


new_data.isnull().sum()


# In[13]:


# Calculate percentage of null values in each column
null_percentage = (new_data.isnull().sum() / len(new_data)) * 100
print("Percentage of Null Values in Each Column:")
print(null_percentage)


# In[14]:


# select the portion of dataframe with no missing values for the column "CustomerID"
clean_data = new_data[pd.notnull(new_data["CustomerID"])]
clean_data.isnull().sum()


# In[15]:


# Calculate total transactions by country
total_transactions_by_country = clean_data.groupby('Country')['InvoiceNo'].count().reset_index()
total_transactions_by_country.columns = ['Country', 'Total Transactions']

# Calculate total transactions across all countries
total_transactions_all_countries = total_transactions_by_country['Total Transactions'].sum()

# Calculate percentage of total transactions for each country
total_transactions_by_country['Percentage of Total Transactions'] = (total_transactions_by_country['Total Transactions'] / total_transactions_all_countries) * 100
print(total_transactions_by_country)


# In[19]:


# Plot the count of transactions by country with log-transformed x-axis
sns.set(style="whitegrid")
plt.figure(figsize=(24, 12))
sns.countplot(data=clean_data, y='Country', palette='viridis')
plt.title('Transactions by Country')
plt.xlabel('Transaction Count (Log Scale)')
plt.ylabel('Country')
plt.xscale('log')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[23]:


# Calculate the distribution of transactions across different countries
transactions_by_country = clean_data['Country'].value_counts()
print("Distribution of Transactions Across Different Countries:")
print(transactions_by_country)

# Calculate the countries with the highest and lowest number of transactions
highest_transactions_country = transactions_by_country.idxmax()
lowest_transactions_country = transactions_by_country.idxmin()
print("\nCountry with the Highest Number of Transactions:", highest_transactions_country)
print("Country with the Lowest Number of Transactions:", lowest_transactions_country)

# Calculate the market penetration as a percentage for each country
total_transactions = len(clean_data)
market_penetration = (transactions_by_country / total_transactions) * 100
print("\nMarket Penetration Across Different Countries:")
print(market_penetration)


# In[26]:


# filter customers from United Kingdom
data = clean_data[clean_data["Country"] == "United Kingdom"]


# In[27]:


data.info()


# In[28]:


data.describe()


# In[29]:


# Calculate the range of dates in the dataset
date_range = pd.to_datetime(data.InvoiceDate.max()) - pd.to_datetime(data.InvoiceDate.min())
print(date_range)


# In[30]:


data.min()
# why will quantity be negative?


# In[31]:


# Filter the DataFrame to show rows where Quantity is negative
negative_quantity_data = data[data['Quantity'] < 0]
print(negative_quantity_data)


# In[32]:


data.max()


# In[33]:


# Filter out returns (transactions with negative quantity)
data = data[data['Quantity'] >= 0]
print(data)


# In[34]:


# Describe the Quantity column
data.Quantity.describe()


# In[35]:


# Create a new column 'Sales' by multiplying 'Quantity' and 'UnitPrice'
data.loc[:, 'Sales'] = data['Quantity'] * data['UnitPrice']
print(data.head())


# In[36]:


# Convert 'InvoiceDate' to datetime format
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')


# In[37]:


# Extract additional date features
data.insert(4, 'Day', data['InvoiceDate'].dt.day)
data.insert(5, 'Month', data['InvoiceDate'].dt.month)
data.insert(6, 'Year', data['InvoiceDate'].dt.year)
data.insert(7, 'WeekDay', data['InvoiceDate'].dt.weekday)
data.insert(8, 'Hour', data['InvoiceDate'].dt.hour)
data.insert(9, 'Minute', data['InvoiceDate'].dt.minute)
data.insert(10, 'Date', data['InvoiceDate'].dt.date)


# In[38]:


data.head()


# In[39]:


# Define a custom color palette
custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Plot the count of transactions by month
sns.catplot(data=data, x='Month', kind='count', palette=custom_palette, height=8.27, aspect=11/8)


# In[40]:


# Plot sales by month
sns.catplot(data=data, x='Month', y='Sales', kind='bar', palette=custom_palette, height=8.27, aspect=11/8)


# In[41]:


sns.catplot(data=data, x= 'WeekDay', y='Sales', kind = 'bar', palette=custom_palette, height=8.27, aspect=11/8)
plt.title('Sales By WeekDay ')
# Monday = 0 to Sunday = 6


# In[42]:


# Plot the count of transactions by hour
sns.catplot(data=data, x='Hour', kind='count', palette=custom_palette, height=8.27, aspect=11/8)


# In[43]:


# Calculate the sum of sales grouped by weekday
sales_by_weekday = data.groupby(['WeekDay']).sum()
sales_by_weekday 


# In[44]:


# Display the counts of each stock code, showing the most frequent ones
stock_code_counts = data['StockCode'].value_counts().head()
stock_code_counts 


# In[45]:


data['CustomerID'].value_counts().head(10)


# In[46]:


data['StockCode'].value_counts().head()


# In[47]:


d_count =  data.Description.value_counts().sort_values(ascending=False).iloc[0:15]
d_count


# In[48]:


# Display the top 15 most frequent product descriptions
top_products = data['Description'].value_counts().sort_values(ascending=False).iloc[0:15]
plt.figure(figsize=(15, 8))
sns.barplot(x=top_products.index, y=top_products.values)
plt.xticks(rotation=90)
plt.title('Top 10 Products')


# ###  Feature Selection (Using Recency, Frequency, and Monetary Value)

# In[50]:


# Calculate RFM Metrics
# Calculate Recency (R)
now = pd.to_datetime('2011-12-09')  # Current date
new_df = data.groupby(by='CustomerID', as_index=False)['InvoiceDate'].max()
new_df.columns = ['CustomerID', 'LastPurchaseDate']
new_df['Recency'] = (now - new_df['LastPurchaseDate']).dt.days
new_df.drop('LastPurchaseDate', axis=1, inplace=True)

# Calculate Frequency (F)
new_df2 = data.groupby(by='CustomerID', as_index=False)['InvoiceNo'].count()
new_df2.columns = ['CustomerID', 'Frequency']

# Calculate Monetary Value (M)
new_df3 = data.groupby(by='CustomerID', as_index=False)['Sales'].sum()
new_df3.columns = ['CustomerID', 'Monetary']


# In[51]:


# Merge RFM Metrics
rfm_df = pd.merge(new_df, new_df2, on='CustomerID')
rfm_df = pd.merge(rfm_df, new_df3, on='CustomerID')
rfm_df.set_index('CustomerID', inplace=True)


# In[52]:


# Merge RFM Metrics
temp = new_df.merge(new_df2, on='CustomerID')
rfm_df = temp.merge(new_df3, on='CustomerID')
rfm_df.set_index('CustomerID', inplace=True)
rfm_df.head(10)


# In[53]:


# Assign RFM Quartiles
rfm_df['R_quartile'] = pd.qcut(rfm_df['Recency'], 4, labels=['1', '2', '3', '4'])
rfm_df['F_quartile'] = pd.qcut(rfm_df['Frequency'], 4, labels=['4', '3', '2', '1'])
rfm_df['M_quartile'] = pd.qcut(rfm_df['Monetary'], 4, labels=['4', '3', '2', '1'])


# In[54]:


# Calculate RFM Score
rfm_df['RFM_Score'] = rfm_df['R_quartile'].astype(str) + rfm_df['F_quartile'].astype(str) + rfm_df['M_quartile'].astype(str)
rfm_df['RFM_Score'] 


# In[55]:


# Explore Segments
# Customers with RFM Score 111 are considered as the best customers
best_customers = rfm_df[rfm_df['RFM_Score'] == '111'].head()
print("Best Customers:")
print(best_customers)


# In[56]:


# Customers with the lowest Frequency and Monetary Value
low_frequency_customers = rfm_df[rfm_df['F_quartile'] == '1'].head()
low_monetary_customers = rfm_df[rfm_df['M_quartile'] == '1'].head()
print("Customers with Lowest Frequency:")
print(low_frequency_customers)
print("Customers with Lowest Monetary Value:")
print(low_monetary_customers)


# In[57]:


# Feature Selection (Using Recency, Frequency, and Monetary Value)
# Selecting features from the RFM DataFrame
features = rfm_df[['Recency', 'Frequency', 'Monetary']]
features.head()


# In[58]:


# Data Preprocessing
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features


# In[59]:


# Customer Segmentation (K-means Clustering) with Elbow Method
# Define a range of clusters to test

k_range = range(1, 11)
wcss = []  # Within-cluster sum of squares

# Calculate WCSS for each k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)  # Inertia is another name for WCSS

# Plot the Elbow Method curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.show()


# In[60]:


# Define a range of clusters to test
k_range = [3, 4, 5, 6]

best_silhouette_score = -1  # Initialize with a negative value
best_kmeans_model = None

for k in k_range:
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    rfm_df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # Evaluate the clustering model using silhouette score
    silhouette_avg = silhouette_score(scaled_features, rfm_df['Cluster'])
    print("For k =", k, "Silhouette Score:", silhouette_avg)
    
    # Save the best model based on silhouette score
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_kmeans_model = kmeans

# Save the Best Model
joblib.dump(best_kmeans_model, 'best_customer_segmentation_model.pkl')


# In[ ]:




