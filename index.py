import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the CSV file into a pandas DataFrame
data = pd.read_csv("./lungDisease.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
data['GENDER'] = label_encoder.fit_transform(data['GENDER'])
data['SMOKING'] = label_encoder.fit_transform(data['SMOKING'])
data['YELLOW_FINGERS'] = label_encoder.fit_transform(data['YELLOW_FINGERS'])
data['ANXIETY'] = label_encoder.fit_transform(data['ANXIETY'])
data['PEER_PRESSURE'] = label_encoder.fit_transform(data['PEER_PRESSURE'])
data['CHRONIC_DISEASE'] = label_encoder.fit_transform(data['CHRONIC_DISEASE'])
data['FATIGUE '] = label_encoder.fit_transform(data['FATIGUE '])
data['ALLERGY '] = label_encoder.fit_transform(data['ALLERGY '])
data['WHEEZING'] = label_encoder.fit_transform(data['WHEEZING'])
data['ALCOHOL CONSUMING'] = label_encoder.fit_transform(data['ALCOHOL CONSUMING'])
data['COUGHING'] = label_encoder.fit_transform(data['COUGHING'])
data['SHORTNESS OF BREATH'] = label_encoder.fit_transform(data['SHORTNESS OF BREATH'])
data['SWALLOWING DIFFICULTY'] = label_encoder.fit_transform(data['SWALLOWING DIFFICULTY'])
data['CHEST PAIN'] = label_encoder.fit_transform(data['CHEST PAIN'])

# Split the data into features (X) and target variable (y)
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
