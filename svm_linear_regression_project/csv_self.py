import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import joblib 
import pandas as pd
import matplotlib.colors as mcolors
#import the dataser from the scv file 

# dataset = pd.read_csv("data/Social_Network_Ads.csv") 
 
# X = dataset.iloc[:, [2, 3]].values
# y = dataset.iloc[:,4].values

# #use the build in function to seperate the train and test dataset 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# #pretrain the model datatset
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# #creat the support vector machine model 
# soft_linear_svm_model = SVC(kernel="rbf", gamma= 10, C = 1e6) 
# #train the svm model 
# soft_linear_svm_model.fit(X_train, y_train)
# #make the prediction q
# y_pred = soft_linear_svm_model.predict(X_test)

# print("size of the y_pred", X_train.shape)

# #draw points 
# x_coor = X_train[:,0]
# Y_coor = X_train[:,1]

# for (index, value) in enumerate(x_coor):
#     x_temp = x_coor[index]
#     y_temp = Y_coor[index]
#     if y_train[index] == 0:
#         plt.scatter(x_temp,y_temp, c='red',s=5)
#     else:
#         plt.scatter(x_temp,y_temp, c='green',s=5)

# # Generate a mesh grid of points spanning the range of the features
# x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
# y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                      np.arange(y_min, y_max, 0.01))

# # Make predictions for each point in the mesh grid
# Z = soft_linear_svm_model.predict(np.c_[xx.ravel(), yy.ravel()])

# # Reshape the predictions to the shape of the mesh grid
# Z = Z.reshape(xx.shape)
# print(Z)
# cmap_custom = mcolors.ListedColormap(['red', 'green'])
# plt.contourf(xx, yy, Z, alpha=0.4,cmap = cmap_custom)
# plt.show()
 



##-------------------------------------------------------
##linear regression 

#save way to get the data from the csv file 
linear_regression_dataset = pd.read_csv("data/Position_Salaries.csv")

#get the x and y coordination 
position_level = linear_regression_dataset.iloc[:,1].values.reshape(-1,1)
salary = linear_regression_dataset.iloc[:,2].values.reshape(-1,1) 
print (position_level.shape)

#we should do the preprocess of the dataset, so that they will have the 0 mean and the 1 sd 
sc_1 = StandardScaler()
sc_2 = StandardScaler()
position_level = sc_1.fit_transform(position_level)
salary = sc_2.fit_transform(salary)

# print ("position_level: ", position_level)
# print ("salary: ", salary)

#inital the linear ergression model 
from sklearn.svm import SVR
svl_model = SVR(kernel='rbf',gamma = 0.1)
svl_model.fit(position_level,salary)

prediction = svl_model.predict(position_level)
plt.scatter(position_level,salary,color='red')
plt.plot(position_level,prediction, color="green")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()








