import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the dataset
ecommerce = pd.read_csv('ecommerce.csv')

# Set Streamlit page config
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Sidebar
#st.sidebar.header('E-commerce Dashboard')
st.sidebar.markdown('<h1 style="color: #1f6f8b;">E-commerce Dashboard</h1>', unsafe_allow_html=True)


# Convert "order_purchase_timestamp" to datetime
ecommerce['order_purchase_timestamp'] = pd.to_datetime(ecommerce['order_purchase_timestamp'])

# Date filter
min_date = ecommerce["order_purchase_timestamp"].min().date()
max_date = ecommerce["order_purchase_timestamp"].max().date()

# Date input
start_date, end_date = st.sidebar.date_input(
    label='Select Date Range',
    min_value=min_date,
    max_value=max_date,
    value=[min_date, max_date]
)

# Filter data based on selected date range
main_df = ecommerce[
    (ecommerce["order_purchase_timestamp"].dt.date >= start_date) &
    (ecommerce["order_purchase_timestamp"].dt.date <= end_date)
]

# Based on the Elbow Method, choose an appropriate k (number of clusters)
chosen_k = st.sidebar.slider('Choose the number of clusters (k):', min_value=2, max_value=6, value=3)

# Display the filtered data
#st.write("Filtered Data:")
#st.write(main_df)

st.caption('Created by Ricky Hermanto')

# Section 1: Total Orders Comparison (January to August)
st.sidebar.subheader('Total Orders Comparison (January to August)')

# Extract year and month from order_purchase_timestamp
ecommerce['order_purchase_year'] = ecommerce['order_purchase_timestamp'].dt.year
ecommerce['order_purchase_month'] = ecommerce['order_purchase_timestamp'].dt.month

# Filter data for the specified time period (January to August) and years (2017 and 2018)
filtered_data = ecommerce[
    (ecommerce['order_purchase_year'].isin([2017, 2018])) & (ecommerce['order_purchase_month'] <= 8)]

# Count of orders per month and year
orders_by_month_year = filtered_data.groupby(['order_purchase_year', 'order_purchase_month']).size().reset_index(
    name='total_orders')

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='order_purchase_month', y='total_orders', hue='order_purchase_year', data=orders_by_month_year,
            palette='viridis')
plt.title('Total Orders Comparison (January to August)')
plt.xlabel('Month')
plt.ylabel('Total Orders')
plt.xticks(ticks=range(1, 9), labels=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August'])
plt.legend(title='Year')

# Calculate total orders for each year
total_orders_2017 = orders_by_month_year[orders_by_month_year['order_purchase_year'] == 2017]['total_orders'].sum()
total_orders_2018 = orders_by_month_year[orders_by_month_year['order_purchase_year'] == 2018]['total_orders'].sum()

# Add a textbox with the total orders for each year
textbox_str = f"Total Orders in 2017: {total_orders_2017}\nTotal Orders in 2018: {total_orders_2018}"
plt.gcf().text(0.1, 0.9, textbox_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

# Add percentage text on top of each bar
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.1, f'{height / len(filtered_data) * 100:.2f}%', ha='center')

# Display the plot in Streamlit
st.pyplot(fig)





# Section 2: Clustering Analysis
# Convert timestamp columns to datetime
ecommerce['order_purchase_timestamp'] = pd.to_datetime(ecommerce['order_purchase_timestamp'])
ecommerce['order_delivered_customer_date'] = pd.to_datetime(ecommerce['order_delivered_customer_date'])

# Calculate delivery time
ecommerce['order_delivery_time'] = (ecommerce['order_delivered_customer_date'] - ecommerce['order_purchase_timestamp']).dt.days

# Select relevant features for clustering
cluster_data = ecommerce[['order_delivery_time', 'review_score']].dropna()

# Standardize the data
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# Determine the optimal number of clusters (k) using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_data_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
fig, ax = plt.subplots()
ax.plot(range(1, 11), inertia, marker='o')
ax.set_title('Elbow Method for Optimal k')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
st.pyplot(fig)

# Apply k-means clustering with the chosen k
kmeans = KMeans(n_clusters=chosen_k, random_state=42)
cluster_data['cluster'] = kmeans.fit_predict(cluster_data_scaled)

# Plot the clusters
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='order_delivery_time', y='review_score', hue='cluster', data=cluster_data, palette='viridis', ax=ax)
ax.set_title('Clustering Analysis')
ax.set_xlabel('Order Delivery Time (days)')
ax.set_ylabel('Review Score')
st.pyplot(fig)



# Section 3: Growing Trend in Brazilian E-commerce
st.sidebar.subheader('Growing Trend in Brazilian E-commerce')

# Extract year and month from order_purchase_timestamp
ecommerce['order_purchase_year_month'] = ecommerce['order_purchase_timestamp'].dt.to_period('M')

# Count of orders per month
order_trend = ecommerce['order_purchase_year_month'].value_counts().sort_index()

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=order_trend.index.astype(str), y=order_trend.values, marker='o', color='skyblue')
plt.title('Growing Trend in Brazilian E-commerce')
plt.xlabel('Year-Month')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
st.pyplot(fig)


# Section 4: Order Frequency by Day of the Week
st.sidebar.subheader('Order Frequency by Day of the Week')

# Extract day of the week from order_purchase_timestamp
ecommerce['order_purchase_day_of_week'] = ecommerce['order_purchase_timestamp'].dt.day_name()

# Count of orders per day of the week
day_of_week_order = ecommerce['order_purchase_day_of_week'].value_counts()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=day_of_week_order.index, y=day_of_week_order.values, palette='viridis')

# Highlight the bar with the highest count
max_count_index = day_of_week_order.idxmax()
for bar in ax.patches:
    if bar.get_height() == day_of_week_order[max_count_index]:
        bar.set_color('orange')  # Change color for the highest bar
    else:
        bar.set_color('lightgray')  # Set other bars to light or blurry color

plt.title('Order Frequency by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Orders')

# Add percentage text on top of the bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(p.get_x() + p.get_width() / 2., height + 0.1, f'{height / len(ecommerce) * 100:.2f}%', ha='center',
                fontsize=8)

st.pyplot(fig)



# Section 5: Order Frequency by Time of Day
st.sidebar.subheader('Order Frequency by Time of Day')

# Define time of day categories
time_of_day_bins = [0, 6, 12, 18, 24]
time_of_day_labels = ['Dawn', 'Morning', 'Afternoon', 'Night']

# Extract hour from order_purchase_timestamp
ecommerce['order_purchase_hour'] = ecommerce['order_purchase_timestamp'].dt.hour

# Categorize the hours
ecommerce['time_of_day'] = pd.cut(ecommerce['order_purchase_hour'], bins=time_of_day_bins, labels=time_of_day_labels,
                                  include_lowest=True)

# Count of orders per time of day
time_of_day_order = ecommerce['time_of_day'].value_counts()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bar_plot = sns.barplot(x=time_of_day_order.index, y=time_of_day_order.values, palette='viridis', ax=ax)

# Highlight the bar with the highest count
max_count_index = time_of_day_order.idxmax()
for bar in bar_plot.patches:
    if bar.get_height() == time_of_day_order[max_count_index]:
        bar.set_color('orange')  # Change color for the highest bar
    else:
        bar.set_color('lightgray')  # Set other bars to light or blurry color

ax.set_title('Order Frequency by Time of Day')
ax.set_xlabel('Time of Day')
ax.set_ylabel('Number of Orders')

# Add percentage text on top of the bars
for p in bar_plot.patches:
    height = p.get_height()
    if height > 0:
        ax.text(p.get_x() + p.get_width() / 2., height + 0.1, f'{height / len(ecommerce) * 100:.2f}%', ha='center',
                fontsize=8)

# Display the plot in Streamlit
st.pyplot(fig)


















# Section 6: Average Order Delivery Time Across Different Regions
# Convert timestamp columns to datetime
ecommerce['order_purchase_timestamp'] = pd.to_datetime(ecommerce['order_purchase_timestamp'])
ecommerce['order_delivered_customer_date'] = pd.to_datetime(ecommerce['order_delivered_customer_date'])

# Calculate delivery time
ecommerce['order_delivery_time'] = (ecommerce['order_delivered_customer_date'] - ecommerce['order_purchase_timestamp']).dt.days

st.sidebar.subheader('Average Order Delivery Time Across Different Regions')
# Plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='customer_state', y='order_delivery_time', data=ecommerce, ci=None, palette='viridis', order=ecommerce['customer_state'].value_counts().index)
plt.title('Average Order Delivery Time Across Different Regions')
plt.xlabel('Customer State')
plt.ylabel('Average Order Delivery Time (days)')
plt.xticks(rotation=45)
st.pyplot(fig)

# Section 7: Count of Order Status
st.sidebar.subheader('Count of Order Status')

# Count of order statuses
order_status_count = ecommerce['order_status'].value_counts()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=order_status_count.index, y=order_status_count.values, palette='viridis')

# Highlight the bar with the highest count
max_count_index = order_status_count.idxmax()
for bar in ax.patches:
    if bar.get_height() == order_status_count[max_count_index]:
        bar.set_color('orange')  # Change color for the highest bar

plt.title('Count of Order Status')
plt.xlabel('Order Status')
plt.ylabel('Count')

# Add percentage text on top of the bars
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.1, f'{height / len(ecommerce) * 100:.2f}%', ha='center')

st.pyplot(fig)

# Section 8: Order Volume Across Different Regions
st.sidebar.subheader('Order Volume Across Different Regions')
# Plot order volume across different regions
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(x='customer_state', data=ecommerce, palette='viridis', order=ecommerce['customer_state'].value_counts().index)
plt.title('Order Volume Across Different Regions')
plt.xlabel('Customer State')
plt.ylabel('Order Count')
plt.xticks(rotation=45)
st.pyplot(fig)


# Section 9: Explore Factors Contributing to Order Frequency
st.sidebar.subheader('Explore Factors Contributing to Order Frequency')

# Factors contributing to order frequency
factors = ['product_category_name', 'payment_type', 'review_score']

for factor in factors:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x=factor, data=ecommerce, palette='viridis', order=ecommerce[factor].value_counts().index)
    plt.title(f'Order Volume by {factor}')
    plt.xlabel(factor.capitalize())
    plt.ylabel('Order Count')
    plt.xticks(rotation=90)
    st.pyplot(fig)

# Distribution of Product Categories
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(x='product_category_name', data=ecommerce, palette='viridis', order=ecommerce['product_category_name'].value_counts().index)
plt.title('Distribution of Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Count')
plt.xticks(rotation=90)
st.pyplot(fig)