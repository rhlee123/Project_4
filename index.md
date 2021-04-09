# GAM vs Nadaraya-Watson Kernel Density Estimation 

Both models are non-parametric contrasting with  techniques such as linear regression which are parametric. Parametric regressions like linear regression incorporate certain assumptions about the data. When a parametric technique is used with data that does not conform to its assumptions, the result of the analysis may be a weak or biased model. Nonparametric regressions on the other hand relaxes assumptions of linearity, enabling the detection and observation of patterns that parametric techniques may miss.

## GAM (General Additive Model)

Generalized Additive Models (GAMs) are an extension of Generalized Linear Models in such a way that predictor variables can be modeled non-parametrically in addition to linear and polynomial terms for other predictors. GAM is a generalized form of the linear model, in which predicted values are predicted using a smoothing functions of the features. GAM blends the properties of generalized linear models and additive models. GAM draws benefits from using principles of generalized linear models while also gaining the benefits of additive models in which simple terms of the linear regression equation can be replaced with more complex smoothing functions.

GAMs are useful when the relationship between the variables are expected to be of a more complex form and not easily fitted by standard linear or non-linear models. The main advantage of GAM is its ability to model highly complex nonlinear relationships when the number of potential predictors (features) is large. Conversely, the main disadvantage of GAM is its computational complexity; like other nonparametric methods, GAM has a high propensity for overfitting.

## Nadaraya-Watson Kernel Density Estimation 

As like any Kernel Density Estimation, Nadaraya-Watson Kernel is a non-paramteric way to estimate the probability density function of a random variable. Further, the Nadaraya-Watson Kernel Density Estimation is a form of kernel density estimation that estimates the predicted values as locally weighted averages through using the Nadaraya-Watson Kernel. 

Lets take a look at how we can derive the Nadaraya-Watson estimator by looking at regression function m. For the case below 𝑋 is the input data (features) and 𝑌 is the dependent variable. If we have estimates for the joint probability density 𝑓 and the marginal density 𝑓𝑋 we consider: 

![image](https://user-images.githubusercontent.com/55299814/114237470-23fb3b00-9951-11eb-9350-60488cd31712.png)

A key takeaway from the expression above is that the regression function can be computed from the joint density 𝑓 and the marginal 𝑓𝑋. So therefore, given a sample (X1,Y1),…,(Xn,Yn), a nonparametric estimate of m may follow by replacing the previous densities with their kernel density estimators. 

We approximate the joint probability density function by using a kernel:

![image](https://user-images.githubusercontent.com/55299814/114237883-b6034380-9951-11eb-950e-f84e455adc41.png)

And further, the marginal density is approximated in a similar way: 

![image](https://user-images.githubusercontent.com/55299814/114237959-d501d580-9951-11eb-9dd0-54eb20dfc8f1.png)

Now we can replace the joint density 𝑓 and the marginal 𝑓𝑋 with the kernel density estimators, to define the estimator of the regression function m:

![image](https://user-images.githubusercontent.com/55299814/114238183-35911280-9952-11eb-9c02-2adba1a8168a.png)

Finally, the resulting estimator is the so-called Nadaraya-Watson estimator of the regression function: 

![image](https://user-images.githubusercontent.com/55299814/114238278-5f4a3980-9952-11eb-91ee-549c5373e663.png)

Where: 

![image](https://user-images.githubusercontent.com/55299814/114238356-79841780-9952-11eb-965c-337b36589d19.png)

Ultimately, Nadaraya–Watson estimator can be seen as a weighted average of Y1,…,Yn by means of the set of weights {Wi(x)}ni=1 (they always add to one). The set of varying weights depends on the evaluation point x. That means that the Nadaraya–Watson estimator is a local mean of Y1,…,Yn about X=x 

# Evaluation 

We will be applying the two different models on our dataset in which we are trying to predict RMSD (size of residue) using the f(1),f(2),f(3),...,f(9) variables as features. To evaluate the performance of the two different models, I looked at each model's Root Mean Squared Error and the coefficient of determination (R^2 value). 

Before we start, the basic imports:
```python 
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as R2
from pygam import LinearGAM
```
Additionally, importing the data as well as data preprocessing shown below:
```python 
df = pd.read_csv('CASP.csv')
features = ['F1','F2','F3','F4','F5','F6','F7','F8','F9']
X = np.array(df[features])
y = np.array(df['RMSD']).reshape(-1,1)
```

Now that the data is preprocessed, let us take a look at the heatmap correlation matrix showing the deeper relationship between the different features we will be using in our model to predict RMSD. The correlation matrix below reveals strong multicolinearity among the different features.

![image](https://user-images.githubusercontent.com/55299814/114244074-cae4d480-995b-11eb-8054-8bd78f2c6164.png)

## GAM: 

Below is the implementation of a KFold function for cross-validating our general additive model (GAM). My GAM was fit with 20 splines in this instance: 
```python 
def DoKFold(X,y,k):
  PE = []
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    gam = LinearGAM(n_splines=20).gridsearch(X_train, y_train,objective='GCV')
    yhat_test = gam.predict(X_test)
    PE.append(math.sqrt(mse(y_test,yhat_test)))
  return 1000*np.mean(PE)
```
Below is the output of using a K-Fold cross validation were k = 10 to evaluate the RMSE of the general additive model (GAM). It can be seen from the elaspsed times that GAM can be a pretty computationally expensive method: 

![image](https://user-images.githubusercontent.com/55299814/114247227-99bbd280-9962-11eb-900a-0b6bed01a64b.png)
