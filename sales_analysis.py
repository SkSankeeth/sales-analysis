# Load Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("supermarket_sales - Sheet1.csv")

# Display initial data overview
print("\nDataset Head:\n", data.head())
print("\nDataset Info:\n")
print(data.info())
print("\n")

# Data Cleaning
# Check for missing values
print("Missing values in each column:\n", data.isnull().sum(), "\n")

# Check for duplicate rows
print("Number of duplicate rows:", data.duplicated().sum(), "\n")

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Convert relevant columns to numeric, ensuring no errors
numeric_columns = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross margin percentage', 'gross income', 'Rating']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Save the cleaned data to a new file
output_file = 'cleaned_supermarket_sales-Sheet1.csv'
data.to_csv(output_file, index=False)
print(f"Cleaned data saved to '{output_file}'\n")

# Exploratory Data Analysis (EDA)
# Display basic statistics
print("Basic Statistics:\n", data.describe())

# Display column-wise data types
print("\nData Types:\n", data.dtypes)

# Visualization
plt.figure(figsize=(10, 6))

# Sales by City
city_sales = data.groupby('City')['Total'].sum()
city_sales.plot(kind='bar', color='skyblue', title='Sales by City')
plt.ylabel('Total Sales')
plt.xlabel('City')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure as an image
plt.savefig('sales_by_city.png')  # Save as PNG image
plt.show()

# Sales by Product Line
product_sales = data.groupby('Product line')['Total'].sum()
product_sales.plot(kind='bar', color='lightgreen', title='Sales by Product Line')
plt.ylabel('Total Sales')
plt.xlabel('Product Line')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure as an image
plt.savefig('sales_by_product_line.png')  # Save as PNG image
plt.show()

# Correlation Heatmap
numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()

# Save the figure as an image
plt.savefig('correlation_heatmap.png')  # Save as PNG image
plt.show()

# Sales Over Time
sales_over_time = data.groupby('Date')['Total'].sum()
sales_over_time.plot(kind='line', color='orange', title='Sales Over Time')
plt.ylabel('Total Sales')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure as an image
plt.savefig('sales_over_time.png')  # Save as PNG image
plt.show()

# Sales by Branch
branch_sales = data.groupby('Branch')['Total'].sum()
branch_sales.plot(kind='pie', autopct='%1.1f%%', title='Sales Distribution by Branch')
plt.ylabel('')  # Hide the y-axis label for clarity

# Save the figure as an image
plt.savefig('sales_by_branch.png')  # Save as PNG image
plt.show()

# Sales by Gender
gender_sales = data.groupby('Gender')['Total'].sum()
gender_sales.plot(kind='bar', color='purple', title='Sales by Gender')
plt.ylabel('Total Sales')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the figure as an image
plt.savefig('sales_by_gender.png')  # Save as PNG image
plt.show()

# Count of Payment Methods
payment_counts = data['Payment'].value_counts()
payment_counts.plot(kind='bar', color='cyan', title='Payment Method Distribution')
plt.ylabel('Count')
plt.xlabel('Payment Method')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the figure as an image
plt.savefig('payment_method_distribution.png')  # Save as PNG image
plt.show()

# Average Rating by Payment Method
payment_rating = data.groupby('Payment')['Rating'].mean()
payment_rating.plot(kind='bar', color='orange', title='Average Rating by Payment Method')
plt.ylabel('Average Rating')
plt.xlabel('Payment Method')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the figure as an image
plt.savefig('average_rating_by_payment_method.png')  # Save as PNG image
plt.show()

# Sales by Product Line
top_products = data.groupby('Product line')['Total'].sum().sort_values(ascending=False)
print("Top Product Lines:\n", top_products)

# Create Average Spend Column
data['Average Spend'] = data['Total'] / data['Quantity']

# Extract Hour from Time
data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour

# Create Spending Categories
conditions = [
    data['Total'] < 200,
    (data['Total'] >= 200) & (data['Total'] < 500),
    data['Total'] >= 500
]
categories = ['Low Spender', 'Medium Spender', 'High Spender']
data['Spender Type'] = pd.cut(data['Total'], bins=[0, 200, 500, float('inf')], labels=categories, right=False)

# Visualize Spending Categories
spender_counts = data['Spender Type'].value_counts()
spender_counts.plot(kind='bar', color='green', title='Spender Type Distribution')
plt.ylabel('Count')
plt.xlabel('Spender Type')
plt.xticks(rotation=0)
plt.show()

# Aggregate Daily Sales
daily_sales = data.groupby('Date')['Total'].sum()

# Plot Daily Sales with Moving Average
daily_sales.plot(label='Daily Sales', color='blue')
daily_sales.rolling(window=7).mean().plot(label='7-Day Moving Average', color='red')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()

# Save the figure as an image
plt.savefig('daily_sales_with_moving_avg.png')  # Save as PNG image
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Prepare the Data
features = pd.get_dummies(data[['Branch', 'Gender', 'Payment']], drop_first=True)
target = data['Total']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the Model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Export Summary Statistics to Excel
summary = data.describe()
summary.to_excel('sales_summary.xlsx')
print("Summary statistics exported to 'sales_summary.xlsx'")
