import pandas as pd
import numpy as np

#ML tools I will use for this
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#load the data from CSV
df = pd.read_csv("gatorade_colors.csv")

#encode gaterade color, assgins each color a number value
le_color = LabelEncoder()
df['color_label'] = le_color.fit_transform(df['gatorade_color'])

#create feature 
df['prev_color']= df['color_label'].shift(1)

#drop first row
df= df.dropna()

#features
X= df[['prev_color']]
y= df['color_label']

#split
X_train, X_test, y_train, y_test,= train_test_split(X,y, test_size=0.2, random_state=42)

#train log reg
model= LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

#evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

#last years data 
prev_color_num = df['color_label'].iloc[-1]  
pred_num = model.predict([[prev_color_num]])
pred_color = le_color.inverse_transform(pred_num)
print("Predicted Gatorade color for 2026:", pred_color[0])
