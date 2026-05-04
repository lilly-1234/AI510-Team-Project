import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("data/customer_support_dataset.csv")

# Show basic information
print("Original dataset shape:", df.shape)
print(df.head())

# Remove missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Convert all text columns to lowercase
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].str.lower()

# Show columns
print("Columns in dataset:")
print(df.columns)

# Save cleaned dataset
df.to_csv("data/cleaned_customer_support_dataset.csv", index=False)
print("Cleaned dataset saved successfully!")

# Choose input and label columns
# For this dataset, usually:
# instruction = customer question
# intent = category label

X = df["instruction"]
y = df["intent"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Show result
print("Training data size:", len(X_train))
print("Testing data size:", len(X_test))
print("Preprocessing completed successfully!")