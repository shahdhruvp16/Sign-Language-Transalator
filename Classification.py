import pickle
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_predict)
conf_matrix = confusion_matrix(y_test, y_predict)
report = classification_report(y_test, y_predict, target_names=[str(i) for i in np.unique(labels)])

# Print metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

# Save confusion matrix as an image
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix.png')
plt.close()

# Generate comparison graph
plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [accuracy * 100], color='skyblue')
plt.title('Model Performance')
plt.ylabel('Percentage')
plt.savefig('model_performance.png')
plt.close()

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)