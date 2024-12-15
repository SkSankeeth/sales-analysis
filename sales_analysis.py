# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("supermarket_sales - Sheet1.csv")

# Display initial overview of the data
print("\nDataset Preview:\n", data.head())
print("\nDataset Information:\n")
print(data.info())
print("\n")

# Data Cleaning
# Check for missing values in each column
print("Missing values per column:\n", data.isnull().sum(), "\n")

# Check for duplicate rows in the dataset
print("Number of duplicate rows:", data.duplicated().sum(), "\n")

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Ensure numeric columns are correctly formatted
numeric_columns = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross margin percentage', 'gross income', 'Rating']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Save the cleaned data to a new CSV file
output_file = 'cleaned_supermarket_sales-Sheet1.csv'
data.to_csv(output_file, index=False)
print(f"Cleaned data saved to '{output_file}'\n")

# Exploratory Data Analysis (EDA)
# Display basic statistics for numeric columns
print("Basic Statistics:\n", data.describe())

# Show the data types of each column
print("\nData Types:\n", data.dtypes)

# Visualization
# Set the figure size for plots
plt.figure(figsize=(10, 6))

# Sales by City
city_sales = data.groupby('City')['Total'].sum()
city_sales.plot(kind='bar', color='skyblue', title='Sales by City')
plt.ylabel('Total Sales')
plt.xlabel('City')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a PNG image
plt.savefig('sales_by_city.png')
plt.show()

# Sales by Product Line
product_sales = data.groupby('Product line')['Total'].sum()
product_sales.plot(kind='bar', color='lightgreen', title='Sales by Product Line')
plt.ylabel('Total Sales')
plt.xlabel('Product Line')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a PNG image
plt.savefig('sales_by_product_line.png')
plt.show()

# Correlation Heatmap of Numeric Data
numeric_data = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()

# Save the plot as a PNG image
plt.savefig('correlation_heatmap.png')
plt.show()

# Sales Over Time (line plot)
sales_over_time = data.groupby('Date')['Total'].sum()
sales_over_time.plot(kind='line', color='orange', title='Sales Over Time')
plt.ylabel('Total Sales')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as a PNG image
plt.savefig('sales_over_time.png')
plt.show()

# Sales Distribution by Branch (pie chart)
branch_sales = data.groupby('Branch')['Total'].sum()
branch_sales.plot(kind='pie', autopct='%1.1f%%', title='Sales Distribution by Branch')
plt.ylabel('')

# Save the plot as a PNG image
plt.savefig('sales_by_branch.png')
plt.show()

# Sales by Gender
gender_sales = data.groupby('Gender')['Total'].sum()
gender_sales.plot(kind='bar', color='purple', title='Sales by Gender')
plt.ylabel('Total Sales')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the plot as a PNG image
plt.savefig('sales_by_gender.png')
plt.show()

# Count of Payment Methods (bar chart)
payment_counts = data['Payment'].value_counts()
payment_counts.plot(kind='bar', color='cyan', title='Payment Method Distribution')
plt.ylabel('Count')
plt.xlabel('Payment Method')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the plot as a PNG image
plt.savefig('payment_method_distribution.png')
plt.show()

# Average Rating by Payment Method (bar chart)
payment_rating = data.groupby('Payment')['Rating'].mean()
payment_rating.plot(kind='bar', color='orange', title='Average Rating by Payment Method')
plt.ylabel('Average Rating')
plt.xlabel('Payment Method')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the plot as a PNG image
plt.savefig('average_rating_by_payment_method.png')
plt.show()

# Sales by Product Line (sorted)
top_products = data.groupby('Product line')['Total'].sum().sort_values(ascending=False)
print("Top Product Lines by Sales:\n", top_products)

# Create a column for Average Spend per item
data['Average Spend'] = data['Total'] / data['Quantity']

# Extract the hour from the 'Time' column
data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour

# Categorize spending levels
conditions = [
    data['Total'] < 200,
    (data['Total'] >= 200) & (data['Total'] < 500),
    data['Total'] >= 500
]
categories = ['Low Spender', 'Medium Spender', 'High Spender']
data['Spender Type'] = pd.cut(data['Total'], bins=[0, 200, 500, float('inf')], labels=categories, right=False)

# Visualize Spending Categories (bar chart)
spender_counts = data['Spender Type'].value_counts()
spender_counts.plot(kind='bar', color='green', title='Spender Type Distribution')
plt.ylabel('Count')
plt.xlabel('Spender Type')
plt.xticks(rotation=0)

# Save the plot as a PNG image
plt.tight_layout()
plt.savefig('spender_type_distribution.png')
plt.show()

# Aggregate Daily Sales (sum by date)
daily_sales = data.groupby('Date')['Total'].sum()

# Plot Daily Sales with 7-Day Moving Average
daily_sales.plot(label='Daily Sales', color='blue')
daily_sales.rolling(window=7).mean().plot(label='7-Day Moving Average', color='red')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()

# Save the plot as a PNG image
plt.savefig('daily_sales_with_moving_avg.png')
plt.show()

# Linear Regression Model for Sales Prediction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Prepare data for the model
features = pd.get_dummies(data[['Branch', 'Gender', 'Payment']], drop_first=True)
target = data['Total']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error of the model: {mse:.2f}")

# Export Summary Statistics to Excel
summary = data.describe()
summary.to_excel('sales_summary.xlsx')
print("Summary statistics exported to 'sales_summary.xlsx'")
