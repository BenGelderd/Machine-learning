"""
The goal for this project is to produce a ML algorithm from scratch utilising Numpy.
This will exclude the use AI entirely and mainly follow apllications taught in University.
This will be combined with the application of cleaned data from SQL.
"""
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


"""
-------------------------------------------
To begin with I will use a simple  weather data set to train, test and later validate my models before applying more advanced 
(and personally manipulated) data sets. 

The first model will be a Linear regression model.

Note: weather_data.csv is a data set taken from Professor John Paul Gosling under the Durham university maths department.

We wish to form the model:

y = Xb + e, where:

y is the (n x 1) vector of response varibale (our desired column)
X is the (n x (p+1)) matrix of respective data plus the intercept
b is the ((p+1) x 1) coefficient vector including the intercept
e is the (n x 1) error vector
--------------------------------------------
"""

Data = pd.read_csv("weather_data.csv")

Data_x = np.array(Data) #Going forward we will want to carry out linear algebra operations on the data set, thus will require array

#Regression model
#-------------------
#Now we address the main concern, linear regression requires numerical data, thus we will manually filter out non-numerical elemnts
Data_x_lm_m = np.delete(Data_x, [4,7,9,10], axis=1)

def combine_int(data): #function to combine intercept to data
    p = data.shape[0] 
    intercept = np.ones((p, 1)) 

    Data_new= np.hstack((intercept, data))
    return(Data_new)



def linear_model(data, beta):
    p, n = data.shape 
    #forming initial vectors for coefficients and noise, add an extra coefficient for the intercept column

    if (beta == 1).all(): #Creating the option to revert to base intercept
        Beta = np.ones(((n + 1), 1)) 
    else:
        Beta = beta

    Epsilon = np.zeros((p, 1))

    Data_x_lm = combine_int(data)

    df = pd.DataFrame(Data_x_lm)
    df.to_csv("lm_Data_new.csv", index=False) #allows us to view our linear model array for any trouble shooting

    Data_y = Data_x_lm @ Beta + Epsilon #Here we have produced our array of predictions given the data above (note what we predict for depends on how we manipulate the data)
    return(Data_y)



"""
-------------------
Plot
Here we will plot the actual data against the predicted where we predict the target variable y by isolating its row.

Our Example here will be predicting the temperature utilising a constant beta = 1, epsilon = 0 and all other numeric variables.
-------------------
"""
print(f"Using beta: {[1, 1, 1, 1, 1, 1, 1]}, to predict the temperature")

y = Data_x_lm_m[:,0].reshape(-1,1)
x = Data_x_lm_m[:,1:]

data_y = linear_model(x, np.array(1)) #Predicted variable list

min_val = min(y.min(), data_y.min())
max_val = max(y.max(), data_y.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Fit")

plt.scatter(y, data_y, alpha = 0.5) 
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Unoptimised predicted temperature")
plt.show()

"""
What this plot tells us is that the regression poorly fits our data, we cannot predict the temperature using these values of
beta and epsilon.
This is why we must optimise our model.

We now introduce the least-squares estimator of beta. b_hat = (X^T X)^(-1)X^Ty. Where y is the known data of the predictor variable.
This is obtained by minimising the sum of squared residuals w.r.t beta, essentially the SSR just tells us how closely our model fits the data.
"""

def min_SSR(y, x):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if not np.all(x[:, 0] == 1):
        x = combine_int(x)

    b_hat = (np.linalg.inv((x.T @ x))) @ (x.T @ y)
    return(b_hat)


#Test on our data above, however need to esnure intercept is included 

Beta_new = min_SSR(y, x)
"""
Having obtained an optimised value for beta we can try the model again and see how it changes
"""
print(f"Using Least Square Estimator of beta: {Beta_new}, to predict the temperature.")

data_y = linear_model(x, Beta_new) #Predicted variable list, issue with our LM function is the change in beta dimensions
plt.scatter(y, data_y, color="blue", alpha=0.5, label="Predictions")

min_val = min(y.min(), data_y.min())
max_val = max(y.max(), data_y.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Optimised prediction")
plt.legend()
plt.show()

#Note: restricted to 100 rows of data for faster processing.

"""
We have built and optimised a Linear regression model and applied it to our numerical data in order to rpedict temperature.

Now we can move on to more accurate predictions by fitting ploynomials to the data, in particular we will try the cubic polynomial.
From there we can begin to calculate percentage errors to draw conclusions between different models.
"""

#We follow the exact same method as our linear model except we need to sum up to the chosen (r) exponent for each variable
#We can do this by forming a new array from the original data raised to the next exponent, then we can combine the data
#into a single array used in the calculation. The rsulting matrix multiplication will carry out the required summation.

def polynomial_model(data, r, beta):
    p, n = data.shape

    if (beta == 1).all(): 
        Beta = np.ones((((r * n) + 1), 1)) #have to scale the coeffeicients due to the exponent increase
    else:
        Beta = beta

    data_new = data
    for i in list(range(2,r+1)):
        x = np.power(data, i)
        data_new = np.hstack((data_new, x))

    Epsilon = np.zeros((p, 1))

    Data_x_lm = combine_int(data_new)

    df = pd.DataFrame(Data_x_lm)
    df.to_csv("lm_Data_new.csv", index=False) #allows us to view our linear model array for any trouble shooting

    Data_y = Data_x_lm @ Beta + Epsilon #Here we have produced our array of predictions given the data above (note what we predict for depends on how we manipulate the data)
    return(Data_y, Data_x_lm) #include manipulated data in output for beta optimisation use

"""
Test and optimise using quadratic and cubic models, once again the initial Beta will be an array of 1s and 0 noise
"""
n=10
print(f"Unoptimised temperature prediction utilising polynoimal of order {n}")

data_y, data_z = polynomial_model(x, n, np.array(1)) #Predicted variable list

plt.scatter(y.flatten(), data_y.flatten(), alpha = 0.5) #ensures we are in one dimension to avoid complications
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title(f"Unoptimised Polynomial Model (Order {n})")
plt.show()

#We see a similar result to our linear model, now lets optimise beta


beta_opt = min_SSR(y, data_z)
data_y_opt , _ = polynomial_model(x, n, beta_opt)

print(f"Temperature prediction using optimised beta: {beta_opt}, order {n}")
plt.scatter(y.flatten(), data_y_opt.flatten(), color="blue", alpha=0.5, label="Predictions")

# Draw the perfect fit line
min_val = min(y.min(), data_y_opt.min())
max_val = max(y.max(), data_y_opt.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Fit")

plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title(f"Optimised Polynomial Model (Order {n})")
plt.legend()
plt.show()

"""
The issue with this temperatur data is that there is 0 correlation and as such no matter the order, the model will only optimise to 
a mean intercept. We can see this with the Beta values, they are extremely small in some cases.

Let us use new data with clear correlation, the anscombe data set from R. This data set is technically four individual data sets combined into one.
"""


# %%


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

#------------------------Copy of above functions----------------------––––––––––¬


def combine_int(data): #function to combine intercept to data
    p = data.shape[0] 
    intercept = np.ones((p, 1)) 

    Data_new= np.hstack((intercept, data))
    return(Data_new)

def min_SSR(y, x):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if not np.all(x[:, 0] == 1):
        x = combine_int(x)

    b_hat = (np.linalg.inv((x.T @ x))) @ (x.T @ y)
    return(b_hat)

def linear_model(data, beta):
    p, n = data.shape 
    #forming initial vectors for coefficients and noise, add an extra coefficient for the intercept column

    if (beta == 1).all(): #Creating the option to revert to base intercept
        Beta = np.ones(((n + 1), 1)) 
    else:
        Beta = beta

    Epsilon = np.ones((p, 1))

    Data_x_lm = combine_int(data)

    df = pd.DataFrame(Data_x_lm)
    df.to_csv("lm_Data_new.csv", index=False) #allows us to view our linear model array for any trouble shooting

    Data_y = Data_x_lm @ Beta + Epsilon #Here we have produced our array of predictions given the data above (note what we predict for depends on how we manipulate the data)
    return(Data_y)

def polynomial_model(data, r, beta):
    p, n = data.shape

    if (beta == 1).all(): 
        Beta = np.ones((((r * n) + 1), 1)) #have to scale the coeffeicients due to the exponent increase
    else:
        Beta = beta

    data_new = data
    for i in list(range(2,r+1)):
        x = np.power(data, i)
        data_new = np.hstack((data_new, x))

    Epsilon = np.zeros((p, 1))

    Data_x_lm = combine_int(data_new)

    df = pd.DataFrame(Data_x_lm)
    df.to_csv("lm_Data_new.csv", index=False) #allows us to view our linear model array for any trouble shooting

    Data_y = Data_x_lm @ Beta + Epsilon #Here we have produced our array of predictions given the data above (note what we predict for depends on how we manipulate the data)
    return(Data_y, Data_x_lm) #include manipulated data in output for beta optimisation use


#----------------------------------------------––––––––––¬

Data = pd.read_csv("Anscombe_data.csv")
Data_x = np.array(Data)

#This data is entirley numeric so does not require manipulation, the main difference is that we have specified our response variables
#Let us observe the response variables in terms of x1.

x = Data_x[:, 0].reshape(-1, 1) #This is vital, forces them to 2D column vectors for our functions to work
y = Data_x[:, 5].reshape(-1, 1)

beta = min_SSR(y, x)
data_y = linear_model(x, beta)
print(f"Linear regression prediction for y1 using x1 data and beta: {beta}")


# 2. Plot the RAW data as dots (scatter doesn't care about order)
plt.scatter(x, y, color="blue", label="Actual Data")

# 3. Create perfectly smooth, sorted X values for the line
# This generates 100 evenly spaced points from the lowest x to the highest x
x_smooth = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)

# 4. Feed the SMOOTH X values into your model to get smooth Y predictions
data_y_smooth = linear_model(x_smooth, beta)

# 5. Plot the smooth line
plt.plot(x_smooth, data_y_smooth, color="red", label="Regression Line")

plt.xlabel("x1")
plt.ylabel("y1")
plt.legend()
plt.show()

n=2
plt.scatter(x, y, color="blue", label="Actual Data")

data_y, data_z = polynomial_model(x, n, np.array([1]))

beta_new = min_SSR(y, data_z)

print(f"Linear regression for order {n} with beta: {beta_new}")

plt.plot(x_smooth, polynomial_model(x_smooth, n, beta_new)[0], color="red", label="Regression line")
plt.xlabel("x1")
plt.ylabel("y1")
plt.legend()
plt.show()


"""
The results of these regression models are much clearer with this data set and yield outcomes that we would expect.

However the framework these these models is currently quite messy and limited to tehir examples. Lets try and "automate"
the process.
"""
# %%
"""
The final plan is to adapt this code into a class which we can call with the data, its specified columns for the predictor
and observer variables as well as the type of regression we wish to carry out.
"""
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

class linear_regression:
    def __init__(self, data, predictor, observer, regression_order, beta_overide = np.array([0])): #If the columns of the observers is >1 then square brackets is required
        self.data = data
        self.predictor = predictor
        self.observer = observer
        self.order = regression_order
        self.beta = beta_overide

        self.y = data[:, predictor].reshape(-1,1) #ensures 2D column vector

        if isinstance(observer, int): #handles the case of 1 vs many columns
            self.x = data[:, observer].reshape(-1, 1)
        else:
            self.x = data[:, observer]


#-----------------regression functions-----------------––––––––––¬

    def combine_int(self, data): #function to combine intercept to data
        p = data.shape[0] 
        intercept = np.ones((p, 1))
        

        Data_new= np.hstack((intercept, data))
        return(Data_new)

    def min_SSR(self, data):
        x = np.asarray(data, dtype=float)
        y = np.asarray(self.y, dtype=float)

        x = self.combine_int(x)


        b_hat = (np.linalg.inv((x.T @ x))) @ (x.T @ y)
        return(b_hat)

    def linear_model(self):
        data = self.x
        p = data.shape[0] 
        #forming initial vectors for coefficients and noise, add an extra coefficient for the intercept column

        if self.beta.all() == np.array([0]):  #Allows for manual beta input (need for training and testing/validating)
            beta = self.min_SSR(self.x)
            beta = beta.reshape(-1,1)

        else:
            beta = self.beta.reshape(-1,1)

        Epsilon = np.ones((p, 1))


        Data_x_lm = self.combine_int(self.x)
        # df = pd.DataFrame(Data_x_lm)
        # df.to_csv("lm_Data_new.csv", index=False) #allows us to view our linear model array for any trouble shooting

        Data_y = Data_x_lm @ beta + Epsilon #Here we have produced our array of predictions given the data above (note what we predict for depends on how we manipulate the data)
        

        if type(self.observer) == int:
            x_smooth = np.linspace(self.x.min(), self.x.max(), 100).reshape(-1,1)
            x_smooth = self.combine_int(x_smooth)
            Epsilon_smooth = np.ones(((x_smooth.shape[0]), 1))
            Data_smooth = x_smooth @ beta + Epsilon_smooth #For smooth output
        else:
            Data_smooth = 0


        self.beta = beta
        return(Data_y, Data_smooth)

    def polynomial_model(self):
        p = self.x.shape[0]

        x_smooth = np.linspace(self.x.min(), self.x.max(), 100).reshape(-1,1)

        data_main = self.x 
        def dim_change(data):
            r = self.order
            data_new = data
            for i in list(range(2, r+1)):
                x = np.power(data, i)
                data_new = np.hstack((data_new, x))
            # data_f = pd.DataFrame(data_new)
            # data_f.to_csv("lm_Data_high_dim.csv", index=False)
            return(data_new)
    

        new_x = dim_change(data_main)
        if self.beta.all() == np.array([0]):
            beta = self.min_SSR(new_x)
        else:
            beta = self.beta.reshape(-1,1)
            
        new_x = self.combine_int(new_x)

        Epsilon = np.ones((p, 1))

        # df = pd.DataFrame(Data_x_lm)
        # df.to_csv("lm_Data_new.csv", index=False) #allows us to view our linear model array for any trouble shooting

        Data_y = new_x @ beta + Epsilon #Here we have produced our array of predictions given the data above (note what we predict for depends on how we manipulate the data)
        if type(self.observer) == int:
            x_smooth = np.linspace(self.x.min(), self.x.max(), 100).reshape(-1,1)
            smooth_new = dim_change(x_smooth)
            x_smooth = self.combine_int(smooth_new)
            Epsilon_smooth = np.zeros(((x_smooth.shape[0]), 1))
            data_y_smooth = x_smooth @ beta + Epsilon_smooth
        else:
            data_y_smooth = 0

        self.beta = beta
        return(Data_y, data_y_smooth) #include manipulated data in output for beta optimisation use
    
    def plot_model_linear(self):
        #We want to consider the data we are trying to display, if the dimensions of the observers is not 2D we will plot predicted vs actual.
        if type(self.observer) == int:
                plt.scatter(self.x, self.y, color="blue", label="Actual Data")
                x_smooth = np.linspace(self.x.min(), self.x.max(), 100).reshape(-1, 1)


                data_y_smooth = self.linear_model()[1]

                # 5. Plot the smooth line
                plt.plot(x_smooth, data_y_smooth, color="red", label="Regression Line")

                plt.xlabel("Observed")
                plt.ylabel("Predicted")
                plt.legend()
                plt.show()


        else:
            data_y = self.linear_model()[0]
            plt.scatter(self.y.flatten(), data_y.flatten(), color="blue", alpha=0.5, label="Predictions")

            min_val = min(self.y.min(), data_y.min())
            max_val = max(self.y.max(), data_y.max())
            plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Fit")

            plt.xlabel("Actual Temperature")
            plt.ylabel("Predicted Temperature")
            plt.title(f"Optimised Linear Model")
            plt.legend()
            plt.show()

    def plot_model_polynoimal(self):
        if type(self.observer) == int:
                plt.scatter(self.x, self.y, color="blue", label="Actual Data")
                x_smooth = np.linspace(self.x.min(), self.x.max(), 100).reshape(-1, 1)


                data_y_smooth = self.polynomial_model()[1]

                # 5. Plot the smooth line
                plt.plot(x_smooth, data_y_smooth, color="red", label="Regression Line")

                plt.xlabel("Observed")
                plt.ylabel("Predicted")
                plt.title(f"Polynomial of {self.order}th order Regression")
                plt.legend()
                plt.show()
        
        else:
            data_y = self.polynomial_model()[0]
            plt.scatter(self.y.flatten(), data_y.flatten(), color="blue", alpha=0.5, label="Predictions")

            min_val = min(self.y.min(), data_y.min())
            max_val = max(self.y.max(), data_y.max())
            plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Fit")

            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(f"Optimised Polynomial Model (Order {self.order})")
            plt.legend()
            plt.show()

#----------------------------------------------––––––––––¬



if __name__ == "__main__":
    run_model = linear_regression(data= np.array(pd.read_csv("Anscombe_data.csv")), predictor=5, observer= 0, regression_order = 4)

    if run_model.order == 1:
        predicted_data = run_model.linear_model()[0]
        run_model.plot_model_linear()
    else:
        predicted_data = run_model.polynomial_model()[0]
        run_model.plot_model_polynoimal()


#%%