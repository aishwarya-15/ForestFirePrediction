import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('C:\\Users\\ELCOT\Downloads\\forest_fire.csv')
print(df)
X = df.drop(['Fire Occurrence','Area'], axis=1).astype(int)
Y = df['Fire Occurrence'].astype(int)
leg = LogisticRegression()
leg.fit(X, Y)
pickle.dump(leg, open('model.pkl', 'wb'))
