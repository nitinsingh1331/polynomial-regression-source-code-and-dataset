#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# In[2]:


df=pd.read_csv("C:/Users/Dell/Downloads/salaries.csv") 
df 


# In[16]:


X=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values 
X 


# In[14]:


from sklearn.linear_model import LinearRegression 
lin_reg=LinearRegression() 
lin_reg.fit(X,y) 


# In[29]:


from sklearn.preprocessing import PolynomialFeatures 
poly_reg=PolynomialFeatures(degree=4) 
X_poly=poly_reg.fit_transform(X) 
X_poly 
lin_reg_2=LinearRegression() 
lin_reg_2.fit(X_poly,y) 


# In[30]:


plt.scatter(X,y,color="blue",marker="*")
plt.plot(X,lin_reg.predict(X),color="red") 
plt.title("truth(LINEAR REGRESSION)")
plt.xlabel("position") 
plt.ylabel("salary") 


# # polynomial  

# In[33]:


plt.scatter(X,y,color="red") 
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color="yellow")
plt.title("truth(polynomial REGRESSION)")
plt.xlabel("position") 
plt.ylabel("salary") 


# In[37]:



x_grid=np.arange(min(X),max(X),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(X,y,color="red") 
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color="yellow")
plt.title("truth(polynomial REGRESSION)")
plt.xlabel("position") 
plt.ylabel("salary")   


# In[38]:


lin_reg.predict([[7.5]]) 


# In[40]:


lin_reg_2.predict(poly_reg.fit_transform([[5.5]]))  

