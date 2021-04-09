# GAM vs Nadaraya-Watson Kernel Density Estimation 

## GAM (General Additive Model)

GAM is a generalized from of the linear model, in which predicted values are predicted using a smoothing functions of the features. GAM blends the properties of generalized linear models and additive models. 

## Nadaraya-Watson Kernel Density Estimation 

Nadaraya-Watson Kernel Density Estimation is a form of kernel density estimation that estimates the predicted values as locally weighted averages through using the Nadaraya-Watson Kernel. 

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

The Nadaraya–Watson estimator can be seen as a weighted average of Y1,…,Yn by means of the set of weights {Wi(x)}ni=1 (they always add to one). The set of varying weights depends on the evaluation point x. That means that the Nadaraya–Watson estimator is a local mean of Y1,…,Yn about X=x 
