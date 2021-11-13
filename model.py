import pandas as pd


from sklearn.model_selection import train_test_split
import pickle


import app
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("diabetes.csv")


x = df[["Pregnancies",	"Glucose",	"BloodPressure",	"SkinThickness",	"Insulin",	"BMI",	"DiabetesPedigreeFunction","Age"]].values
y = df["Outcome"].values

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.1, random_state = 0)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

pred_y = classifier.predict(x_test)
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,pred_y)
print(ac)
print(pred_y)
pickle.dump(classifier, open("model.pkl","wb"))
model = pickle.load(open("model.pkl","rb"))

