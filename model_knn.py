import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("mol_desc_given.csv", index_col=0)

X = df[set(df.columns) - set(['ALDH1_inhibition'])]
y = df['ALDH1_inhibition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)
