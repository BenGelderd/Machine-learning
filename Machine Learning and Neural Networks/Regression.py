#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

"""
We have created a linear regression algorithm, now we will create more advanced regularised regression algorithms such as Ridge, Lasso
and Elastic net.
Data needs to be standardised.
"""
np.random.seed(123)

data = pd.read_csv("iris.csv")

cols_vector = data.columns.to_numpy()

data = np.array(data)

train_size = int(data.shape[0] * 0.75)

train_indices = np.random.choice(data.shape[0], size=train_size, replace=False)
test_indices = np.setdiff1d(np.arange(data.shape[0]), train_indices) #Useful function that takes the array not included above

train_set = data[train_indices,:]
test_set = data[test_indices,:]

"""
We will use similar functions as to the previous regression class but slightly modified.
"""


class Regression:
    def __init__(self, data, observer, predictor, lambd, order = 1, epsilon = 0, alpha=0.5, initial_beta = np.array([1])):
        self.obs = observer
        self.pred = predictor
        self.lambd = lambd
        self.alpha = alpha
        self.beta = initial_beta
        self.epsilon = epsilon
        self.order = order

        if type(self.obs) == int: #allow for multidimensional input
                self.data = data[:, [self.obs, self.pred]]
        
        else:
            index = self.obs + [self.pred]
            self.data = data[:, index]

        self.pred_pos = (self.data.shape[1] - 1)
        self.obs_pos = np.arange(self.pred_pos)

        #A bit tedious but required for polynomail models.
        self.pred_pos_original = self.pred_pos
        self.obs_pos_original = self.obs_pos

        if self.order > 1:
                x = self.data[:, self.obs_pos]
                y = self.data[:, self.pred_pos].reshape(-1, 1)
                
                x_poly = self.create_poly_features(x)
                self.data = np.hstack((x_poly, y))
                
                # Update positions because we just added a bunch of columns!
                self.pred_pos = (self.data.shape[1] - 1)
                self.obs_pos = np.arange(self.pred_pos)
        
        #Store the purely RAW, unscaled data safely
        self.raw_data = np.asarray(self.data, dtype=float)
        
        #Initialize placeholders for our scaling metrics
        self.mean_vals = None
        self.std_vals = None
        self.data = None # This will hold the scaled "working" data
        
        #Scale the full dataset initially (in case the user skips K-fold CV)
        self.scale_and_set_data(self.raw_data)

    def scale_and_set_data(self, raw_data_subset):
        """
        Calculates mean and std on the provided subset ONLY, 
        and updates self.data with the scaled version.
        """
        self.mean_vals = np.mean(raw_data_subset, axis=0)
        self.std_vals = np.std(raw_data_subset, axis=0) + 1e-8
        
        self.data = (raw_data_subset - self.mean_vals) / self.std_vals


    def create_poly_features(self, x):
            x_poly = x.copy()
            for i in range(2, self.order + 1):
                x_poly = np.hstack((x_poly, np.power(x, i)))
            return(x_poly)

    def combine_int(self, data): #function to combine intercept to data
            p = data.shape[0] 
            intercept = np.ones((p, 1))
            
    
            Data_new= np.hstack((intercept, data))
            return(Data_new)
    
    def min_SSR(self):
        x = np.asarray(self.data[:,self.obs_pos], dtype=float)
        y = np.asarray(self.data[:,self.pred_pos], dtype=float).reshape(1,-1)
    
        x = self.combine_int(x)
    
    
        b_hat = (np.linalg.inv((x.T @ x))) @ (x.T @ y)
        return(b_hat)


    def Ridge(self):
        x = np.asarray(self.data[:,self.obs_pos], dtype=float)
        x = self.combine_int(x)
        y = np.asarray(self.data[:,self.pred_pos], dtype=float)

        I = np.eye(x.shape[1])
        I[0,0] = 0 #dont want lambda to act on the intercept

        n = self.data.shape[0]

        b_hat = np.linalg.inv((x.T @ x) + (n*self.lambd*I)) @ (x.T @ y)
        self.beta = b_hat
        # print(f"Ridge beta value of {b_hat}")
        return(b_hat)


    def lasso(self, iterations = 1000): #requires iteration in order to update beta to the optimal coefficient vector, most libraries will use coordinate descent
        # we will use this process however to save time will obtain the nested loop from AI.
        x = self.combine_int(self.data[:,self.obs_pos])

        self.beta = np.zeros(x.shape[1])

        # Coordinate Descent Loop
        for _ in range(iterations):
            for j in range(x.shape[1]):
                # Temporarily remove the current feature's weight
                tmp_weights = self.beta.copy()
                tmp_weights[j] = 0.0
                
                #  Calculate the residual error
                l = self.data[:,self.pred_pos] - (x @ tmp_weights)
                
                # Calculate the correlation between this feature and the error (determining how useful the feature if for the regression)
                rho = (x[:, j] @ l) / x.shape[0]
                
                # The Soft Thresholding solution
                if j ==0:
                     self.beta[j] = rho
                elif rho < -self.lambd:
                    self.beta[j] = rho + self.lambd
                elif rho > self.lambd:
                    self.beta[j] = rho - self.lambd
                else:
                    self.beta[j] = 0.0 # Snaps to zero

        b_hat = self.beta
        # print(f"Lasso beta value of {b_hat}")
        return(b_hat)

    def elastic_net(self, iterations=1000):
        x = self.combine_int(self.data[:, self.obs_pos])
        self.beta = np.zeros(x.shape[1])
        n_samples = x.shape[0]


        l1 = self.lambd * self.alpha
        l2 = self.lambd * (1 - self.alpha)

        # Coordinate Descent Loop again
        for _ in range(iterations):
            for j in range(x.shape[1]):
                tmp_weights = self.beta.copy()
                tmp_weights[j] = 0.0
                
                l = self.data[:, self.pred_pos] - (x @ tmp_weights)
                rho = (x[:, j] @ l) / n_samples
                
                if j == 0:
                    self.beta[j] = rho 
                else:
                    if rho < -l1:
                        beta_j = rho + l1
                    elif rho > l1:
                        beta_j = rho - l1
                    else:
                        beta_j = 0.0 
                        
                    self.beta[j] = beta_j / (1.0 + l2)

        b_hat = self.beta
        # print(f"Elastic Net beta value of {b_hat}")
        return(b_hat)

    def predict(self, x):
        x = self.combine_int(x)
        return(x @ self.beta + self.epsilon)


    def plot_model(self, data=None):
        if data is None:
            data = self.data
            
        #Determine if we are looking at scaled internal data, or raw external data for when apply to test data (idea taken from AI)
        is_internal = np.array_equal(data, self.data)
            
        if type(self.obs) == int:
            if is_internal:
                # internal data is scaled, and Y has been pushed to the end (self.pred_pos)
                x = (data[:, 0] * self.std_vals[0]) + self.mean_vals[0]
                y = (data[:, self.pred_pos] * self.std_vals[self.pred_pos]) + self.mean_vals[self.pred_pos]
            else:
                # test data is completely raw and unscaled
                x = data[:, self.obs_pos_original]
                y = data[:, self.pred_pos_original]

            plt.scatter(x, y, color="blue", label="Actual Data")

            x_smooth = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)

            # Expanding and scaling the smooth line for predictions
            if self.order > 1:
                x_smooth_poly = self.create_poly_features(x_smooth)
                x_smooth_scaled = (x_smooth_poly - self.mean_vals[self.obs_pos]) / self.std_vals[self.obs_pos]

            else:
                x_smooth_scaled = (x_smooth - self.mean_vals[self.obs_pos]) / self.std_vals[self.obs_pos]

            data_y_smooth_scaled = self.predict(x_smooth_scaled)
            
            # Unscaling the predictions back to realistic data
            data_y_smooth = (data_y_smooth_scaled * self.std_vals[self.pred_pos]) + self.mean_vals[self.pred_pos]

            plt.plot(x_smooth, data_y_smooth, color="red", label="Regression Line")
            plt.xlabel("Observed")
            plt.ylabel("Predicted")
            plt.legend()
            plt.show()

        else:
            if is_internal:
                x_input = data[:, self.obs_pos]
                y_actual_scaled = data[:, self.pred_pos]
            else:
                x_raw = data[:, self.obs_pos_original]
                y_raw = data[:, self.pred_pos_original]

                if self.order > 1:
                    x_input = self.create_poly_features(x_raw)
                else:
                    x_input = x_raw

                x_input = (x_input - self.mean_vals[self.obs_pos]) / self.std_vals[self.obs_pos]
                y_actual_scaled = (y_raw - self.mean_vals[self.pred_pos]) / self.std_vals[self.pred_pos]

            data_y_scaled = self.predict(x_input)
            
            data_y = (data_y_scaled * self.std_vals[self.pred_pos]) + self.mean_vals[self.pred_pos]
            y = (y_actual_scaled * self.std_vals[self.pred_pos]) + self.mean_vals[self.pred_pos]
            
            plt.scatter(y.flatten(), data_y.flatten(), color="blue", alpha=0.5, label="Predictions")

            min_val = min(y.min(), data_y.min())
            max_val = max(y.max(), data_y.max())
            plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Fit")

            plt.xlabel("Actual Value")
            plt.ylabel("Predicted Value")
            plt.legend()
            plt.show()

    def mse(self, x, y):
        if self.order > 1:
            x_input = self.create_poly_features(x)
        else:
            x_input = x

        #Need to scale the input features using the training mean/std
        x_scaled = (x_input - self.mean_vals[self.obs_pos]) / self.std_vals[self.obs_pos]

        
        y_pred_scaled = self.predict(x_scaled)

        #Unscale the predictions
        y_pred = (y_pred_scaled * self.std_vals[self.pred_pos]) + self.mean_vals[self.pred_pos]
        
        return np.mean((y - y_pred.flatten()) ** 2)

    """
    Let us now emplopy hyperparamter selection to find the optimal lambda and alpha vlaues, in particular we will use K fold cross validation.
    """
    def K_fold_CV(self, method, k=11): #new version that prevents data leakage by scaling the 
        #training fold only and then applying those metrics to the validation fold.

        # Slice from the RAW data to prevent leakage
        original_raw_data = self.raw_data.copy() 
        n = original_raw_data.shape[0]
        
        indices = np.random.permutation(n)
        folds = np.array_split(indices, k)

        hyp_cand_lamb = np.linspace(0.001, 1, 5) # [0.001, 0.25075, 0.5005, 0.75025, 1. ]
        hyp_cand_lamb = np.append(hyp_cand_lamb, [0.0, 10.0])
        hyp_cand_alpha = np.linspace(0.00001, 1, 20)

        mse_scores = {(l, a): [] for l in hyp_cand_lamb for a in hyp_cand_alpha}

        for valid_idx in folds:
            train_idx = np.setdiff1d(indices, valid_idx)
            
            for l in hyp_cand_lamb:
                for a in hyp_cand_alpha:
                    self.lambd = l
                    self.alpha = a
                    
                    # 1. SCALE TRAINING FOLD ONLY
                    # This safely updates self.mean_vals and self.std_vals without seeing valid data
                    train_fold_raw = original_raw_data[train_idx]
                    self.scale_and_set_data(train_fold_raw)
                    
                    # 2. Fit the model on the scaled training data
                    if method in ["EN", "elastic net", "Elastic net", "en", "Elastic Net"]:
                        self.elastic_net()
                    elif method in ["Ridge", "ridge"]:
                        self.Ridge()
                    else:
                        self.lasso()

                    # 3. SCALE VALIDATION FOLD USING TRAINING METRICS
                    valid_fold_raw = original_raw_data[valid_idx]
                    valid_fold_scaled = (valid_fold_raw - self.mean_vals) / self.std_vals
                    
                    # Temporarily point self.data to the scaled validation set for prediction
                    self.data = valid_fold_scaled 
                    y_actual = self.data[:, self.pred_pos]
                    y_valid_pred = self.predict(self.data[:, self.obs_pos])
                    
                    fold_mse = np.mean((y_actual - y_valid_pred) ** 2)
                    mse_scores[(l, a)].append(fold_mse)

                    if method not in ["EN", "elastic net", "Elastic net", "en", "Elastic Net"]:
                        break
        
        # Find the best hyperparameters
        best_lambda, best_alpha = None, None
        min_avg_mse = float('inf')

        for (l, a), scores in mse_scores.items():
            if not scores:
                continue

            avg_mse = np.mean(scores)
            if avg_mse < min_avg_mse:
                min_avg_mse = avg_mse
                best_lambda = l
                best_alpha = a
        
        self.lambd = best_lambda
        self.alpha = best_alpha
        
        # 4. FINAL REFIT
        # Re-scale using the ENTIRE training dataset before the final fit
        self.scale_and_set_data(original_raw_data)
        
        if method in ["EN", "elastic net", "Elastic net", "en", "Elastic Net"]:
            self.elastic_net()
            print(f"Optimal lambda, alpha for {method} = {best_lambda:.4f}, {best_alpha:.4f}")
        elif method in ["Ridge", "ridge"]:
            self.Ridge()
            print(f"Optimal lambda for {method} = {best_lambda:.4f}")
        else:
            self.lasso()
            print(f"Optimal lambda for {method} = {best_lambda:.4f}")
        
    
if __name__ == "__main__":
    my_model = Regression(data=train_set, epsilon=1e-3, order=1, predictor=3, observer=1, lambd=0.5, alpha=0.5)                            
    my_model.K_fold_CV(method="EN", k=5)
    
    my_model.plot_model()

    print("test data fitting")
    
    if isinstance(my_model.obs, int):
        X_test = test_set[:, my_model.obs].reshape(-1, 1)  # Force 1D into 2D
    else:
        X_test = test_set[:, my_model.obs] 

    y_test = test_set[:, my_model.pred]
    
    print(f"test data MSE: {my_model.mse(X_test, y_test):.4f}")
    
    # Note: Your plot_model is written very cleverly to plot "Actual vs Predicted" 
    # when self.obs is a list, so this will work perfectly for multiple dimensions too!
    
    #my_model.plot_model(data=test_set)



    # %%


 # def K_fold_CV(self, method, k=5): (original data leakage version)
    #     original_data = self.data
    #     #begin with splitting data into train and validation folds for each hyper paramater candidate.
    #     n = self.data.shape[0]
    #     indices = np.random.permutation(n)
    #     size = n // k

    #     hyp_cand_lamb = np.linspace(0,1000, 20)
    #     hyp_cand_alpha = np.linspace(0.00001,1, 20)

    #     MSE = []
    #     for i in range(k):
    #         start = size * i
    #         if i == k-1:
    #             end = n
    #         else:
    #             end = start + size

    #         valid = indices[start:end]
    #         train = np.setdiff1d(indices, valid)

            
    #         for l in hyp_cand_lamb:
    #             for j in hyp_cand_alpha:
    #                 self.data = self.data[train]
    #                 self.lambd = l
    #                 self.alpha = j
                    
    #                 if method in ["EN", "elastic net", "Elastic net", "en", "Elastic Net"]:
    #                     self.elastic_net()

    #                 elif method in ["Ridge", "ridge"]:
    #                     self.Ridge()

    #                 else:
    #                     self.lasso()

    #                 self.data = original_data[valid]
    #                 y = self.data[:,self.pred_pos]
    #                 valid_result = self.predict(self.data[:,self.obs_pos])
    #                 MSE.append((np.mean((y - valid_result)**2), l, j))
    #                 self.data = original_data

    #     self.data = original_data
    #     all_mse = [i[0] for i in MSE]
    #     min_index = np.argsort(all_mse)
    #     i = MSE[min_index[0]][1]
    #     j = MSE[min_index[0]][2]
    #     print(f"Optimal lambda, alpha for {method} = {i},{j}")
    #     print(f"Setting lambda and alpha to {i} and {j}")
    #     self.lambd = i
    #     self.alpha = j
    #     return