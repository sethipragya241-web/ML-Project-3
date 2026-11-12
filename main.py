import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df= pd.read_csv("Student_Performance_DT (1).csv")
print(df.head)
print(df.isna().sum())
df['Internet_Access']= df['Internet_Access'].map({"Yes":0, "No":1})
df['Extracurricular']= df['Extracurricular'].map({"Yes":0, "No":1})
x=df[['Internet_Access','Extracurricular', 'Assignments_Completed','Previous_Score','Participation','Sleep_Hours','Attendance','Study_Hours']]
y=df[['Final_Result']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)
trad_model = DecisionTreeClassifier(random_state=42)
trad_model.fit(x_train, y_train)
y_pred = trad_model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("accuracy of traditional model", acc)