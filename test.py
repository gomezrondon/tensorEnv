import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
# import matplotlib.style import style


data = pd.read_csv("student-mat.csv",sep=";")
# print(data.head())
data = data[["G1","G2","G3","studytime","failures","absences"]]
# print(data)

predict = "G3" # utilizando las demas columnas voy a predecir G3
x = np.array(data.drop([predict],1)) # datos de entrenamiento
y = np.array(data[predict]) #el valor que quiero predecir

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# x_train y y train son 90% de los datos dedicados a entrenar
# 0.1 = 10% de los datos es para las pruebas del modelo final
# no se le debe dara la misma data para entrenar y para probar, ya que la memoriza

#creamos el modelo
# linear = linear_model.LinearRegression()
# linear.fit(x_train, y_train)# entrenamos el modelo
#
# acc = linear.score(x_test, y_test)
# print(acc)
#
# #------------------------------------------------
# with open("student_model","wb") as f: # guarda en un archivo el modelo entrenado
#     pickle.dump(linear, f)

modelo = open("student_model","rb") # cargamos el modelo guardado
linear = pickle.load(modelo)
#------------------------------------------------


print("Co:",linear.coef_)# constantes
print("Intercept: ",linear.intercept_)

#como probamos esto??
predictions = linear.predict(x_test) # probamos el modelo creado
for x in range(len(predictions)):
    print("prediccion: ",round(predictions[x],1),"test data:",x_test[x], "Resultado verdadero:",y_test[x])

# como plotear el modelo