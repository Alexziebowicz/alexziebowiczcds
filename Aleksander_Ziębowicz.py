import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt



#1
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
columns = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
'Horizontal_Distance_To_Fire_Points']+['Wilderness_Area_'+ str(i) for i in range(4)] +['Soil_Type_' + str(i) for i in range(40)]+ ['Cover_Type']
df = pd.read_csv(url, compression='gzip',  names=columns)

print(df)

#2
def split_data(df,Elevation ):
    set1 = df[df['Elevation'] >= 3000]
    set2 = df[df['Elevation'] < 3000]
    return set1, set2

set1, set2 = split_data(df, 3000)

print("Set1:", len(set1),"Set2:", len(set2))

#3
#Linear Regression
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Root Mean Squared Error (test): {score:}")

#Decision trees
X_train2, X_test2, y_train2, y_test2 = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

y_prediction = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_prediction)
print(f"Accuracy: {accuracy}")

#4
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values - 1 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(hidden_layer_sizes, dropout_rate, learning_rate):
    model = tf.keras.models.Sequential()
    for size in hidden_layer_sizes:
        model.add(tf.keras.layers.Dense(size, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(7, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

param_dist = {'hidden_layer_sizes': [(64,), (128,), (256,), (64, 64), (128, 64)],'dropout_rate': [0.0, 0.1, 0.2, 0.3],
              'learning_rate': [0.001, 0.01, 0.1]}
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=50, batch_size=256, verbose=0)
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
best_model = random_search.best_estimator_
history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=256, verbose=0)

#5
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

