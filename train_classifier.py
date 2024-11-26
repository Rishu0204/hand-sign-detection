import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

cleaned_data = []
invalid_indices = []

for i, d in enumerate(data_dict['data']):
    if isinstance(d, list) and len(d) == 42:
        cleaned_data.append(d)
    else:
        invalid_indices.append(i)  # Track invalid indices
        print(f"Skipping invalid element at index {i}: {d}")

data = np.array(cleaned_data, dtype='float32')
labels = np.array(data_dict['labels'][:len(cleaned_data)])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()