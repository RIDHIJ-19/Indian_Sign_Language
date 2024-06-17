import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Ensure all data entries are of the same length
data = data_dict['data']
labels = data_dict['labels']

# Determine the maximum length of the data entries
max_length = max(len(entry) for entry in data)
print(f"Max length of data entries: {max_length}")

# Pad sequences to ensure uniform shape
def pad_sequence(sequence, max_length):
    return np.pad(sequence, (0, max_length - len(sequence)), mode='constant')

# Apply padding to all data entries
padded_data = np.array([pad_sequence(entry, max_length) for entry in data])
labels = np.array(labels)

print(f"Shape of padded data: {padded_data.shape}")

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100}% of samples were classified correctly!')

# Save the trained model and max_length
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'max_length': max_length}, f)
