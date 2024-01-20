import pandas as pd 

import numpy as np 

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import accuracy_score, confusion_matrix 

 
 

# Step 1: Load and Normalize the Dataset 

data = pd.read_csv('wdbc.data.mb.csv', header=None)  # Assuming no header 

X = data.iloc[:, :-1].values  # Features 

y = data.iloc[:, -1].values   # Class labels 

 
 

# Normalize the dataset using Z-score normalization 

scaler = StandardScaler() 

X_normalized = scaler.fit_transform(X) 

 
 

# Step 2: Split the Dataset 

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42) 

 
 

# Step 3 and 4: Distance Calculation and Assignment Modules (Euclidean distance) 

def euclidean_distance(point1, point2): 

    return np.sqrt(np.sum((point1 - point2) ** 2)) 

 
 

def assign_class(test_point, k, X_train, y_train): 

    distances = [(euclidean_distance(test_point, X_train[i]), y_train[i]) for i in range(len(X_train))] 

    sorted_distances = sorted(distances, key=lambda x: x[0]) 

    k_nearest_neighbors = [x[1] for x in sorted_distances[:k]] 

    return max(set(k_nearest_neighbors), key=k_nearest_neighbors.count) 

 
 

# Step 5: Implement kNN 

def kNN(X_train, y_train, X_test, k_values): 

    predictions = {} 

    for k in k_values: 

        y_pred = [assign_class(test_point, k, X_train, y_train) for test_point in X_test] 

        predictions[k] = y_pred 

    return predictions 

 
 

# Step 6: Evaluate and Display Results for different k values 

k_values = [1, 3, 5, 7, 9] 

results = kNN(X_train, y_train, X_test, k_values) 

 
 

for k, y_pred in results.items(): 

    accuracy = accuracy_score(y_test, y_pred) 

    confusion = confusion_matrix(y_test, y_pred) 

    print(f"Results for k={k}:") 

    print(f"Accuracy: {accuracy}") 

    print("Confusion Matrix:") 

    print(confusion) 

    print() 