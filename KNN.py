import numpy as np
import pandas as pd
from sklearn import preprocessing,model_selection,neighbors

df=pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -9999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
df.head()

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])
X=preprocessing.scale(X)
print(len(X),len(y))

X_train,X_test, y_train, y_test=model_selection.train_test_split(X,y,test_size=0.2)
clf=neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
#%%%
accuracy=clf.score(X_test,y_test)
print(accuracy)
