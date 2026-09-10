#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(141)

data = pd.read_csv("iris.csv")

test_data = np.array([6, 2.8, 1.8, 2.5, "Virginica"])



cols_vector = data.columns.to_numpy()

data = np.array(data)

train_size = int(data.shape[0] * 0.75)

train_indices = np.random.choice(data.shape[0], size=train_size, replace=False)
test_indices = np.setdiff1d(np.arange(data.shape[0]), train_indices) #Useful function that takes the array not included above

train_set = data[train_indices,:]
test_set = data[test_indices,:]

class decision_tree:
    def __init__(self, data, observer, predictor, max_features=5 ,max_depth = 3):
        self.data = data
        self.mf = max_features
        self.depth = max_depth
        self.obs = observer
        self.pred = predictor


        if type(self.obs) == int: #allow for multidimensional input
            self.data = data[:, [self.obs, self.pred]]
                
        else:
            index = self.obs + [self.pred]
            self.data = data[:, index]

        self.pred_pos = (self.data.shape[1] - 1)
        self.obs_pos = np.arange(self.pred_pos)
        self.data_original = self.data

    def gini(self, data): #Use gini impurity to split data, require an update to self.data after each node.
        classes = np.unique(self.data_original[:, self.pred_pos])
        c = len(classes)
        n = data.shape[0]
        if n == 0:
            return 0.0
        
        total_prob = np.zeros((c,1))

        for j,i in enumerate(classes):
            count = data[data[:,(data.shape[1]-1)] == i]
            c = count.shape[0]
            prob = c/n
            total_prob[j] = prob**2

        impurity = 1 - np.sum(total_prob)
        return(impurity)

    def majority(self, data):
        classes = np.unique(self.data_original[:, self.pred_pos])
        counts = []
        for c in classes:
            to_count = data[data[:, data.shape[1]-1] == c]
            count = to_count.shape[0]
            counts.append(count)
        pos = np.argsort(counts)

        majority_class = classes[pos][-1]

        return(majority_class)

    def node_split(self, data): #need to calculate the parent impurity, then for each possible observation split, calculate the child impurity
        # the greatest difference represents greatest information gain and we choose that split. Does this for all observation columns before deciding the split.
        parent_gini = self.gini(data)
        features = cols_vector[self.obs]

        parent_data = data


        ideal_split = parent_data.shape[0] /2
        
        inf_array = np.zeros((len(features),4), dtype=object)

        for c,i in enumerate(features):
            self.data = parent_data[:, [c,(parent_data.shape[1]-1)]]
            total = self.data.shape[0]
            inf_gain = -1
            split_val = 0
            diff = ideal_split

            sort_order = np.argsort(self.data[:, 0]) #need to order data of the feature column
            self.data = self.data[sort_order]

            for n in np.arange(self.data.shape[0]):
                left_data = self.data[:(n+1)]
                right_data = self.data[(n+1):]

                impurity = ((left_data.shape[0] / total) * self.gini(left_data)) + ((right_data.shape[0] / total) * self.gini(right_data))

                inf_gain_new = parent_gini - impurity

                if inf_gain_new > inf_gain:
                    inf_gain = inf_gain_new
                    split_val = self.data[n, 0]
                    inf_array[c] = [i, inf_gain, split_val, (len(left_data))]

                elif inf_gain_new == inf_gain:
                    difference = abs(ideal_split - left_data.shape[0])
                    if difference < diff:
                        inf_gain = inf_gain_new
                        split_val = self.data[n, 0]
                        inf_array[c] = [i, inf_gain, split_val, (len(left_data))]
                    else:
                        pass

        self.data = self.data_original

        # print(f"Best split values: {inf_array}")
        compare = [i[1] for i in inf_array]
        max_val = max(compare)
        max_result = [c for c in inf_array if c[1] == max_val]
         #check to see if there are any ties, if so we will choose the one with the most symmetrical split for cleaner branches

        if len(max_result) != 1:
            # print("Finding best symmetry")
            splits = np.array([float(i[3]) for i in max_result])
            diffs = np.abs(ideal_split - splits)
            min_index = np.argmin(diffs)
            final_split = max_result[min_index]

        else:
            final_split = max_result[0]

        #order orginal complete data in terms of the split feature then isolate the varibales for each branch. 
        split_feature = final_split[0]
        feature_location = np.where(cols_vector == split_feature)[0][0]

        isolated_feature = data[:,feature_location]
        order = np.argsort(isolated_feature)
        ordered_data = data[order]

        left_data = ordered_data[ordered_data[:,feature_location] <= float(final_split[2])]

        right_data = ordered_data[ordered_data[:,feature_location] > float(final_split[2])]
        
        
        # print("------------------------------------------------------------------")
        # print(f"Final split chosen as: {final_split[[0,2]]}")
        return([final_split[0], final_split[2]], left_data, right_data)


    def predict(self, test):
        """
        Navigates a single row of test data through the saved tree_map.
        """
        node = self.tree_map
        
        # Keep looping until we hit a dictionary that has a leaf node
        while 'class' not in node:
            feature_name = node['feature']
            threshold = node['threshold']
            
            feature_loc = np.where(cols_vector == feature_name)[0][0]
            
            # Follow the path
            if float(test[feature_loc]) <= threshold:
                node = node['left']
            else:
                node = node['right']
                
        # Return the final predicted class!
        return node['class']


    def decisiontree(self):
        data_matrix = [self.data]

        #creating a dictionary to allow for test data results
        self.tree_map = {}
        dict_matrix = [self.tree_map]


        results_array = []
        n = 0
        final_array = []

        while n < self.depth:
            next_layer = []

            next_dict_layer = []

            if len(data_matrix) == 0:
                break

            print(f"-------------------------{n+1}{'st' if (n+1) % 10 == 1 else 'nd' if (n+1) % 10 == 2 else 'rd' \
                                                if (n+1) % 10 == 3 else 'th'} layer -----------------------------------")
            for x,i in enumerate(data_matrix): #initial start

                current_dict = dict_matrix[x]

                split_parent = self.node_split(i)

                print(f"majority class of {"first" if n == 0 else "left" if (x % 2) == 0 else "right"} node: {self.majority(data=i)}")

                left_data = split_parent[1]
                right_data = split_parent[2]

                # Guardrail 1: Pure Node
                if len(i) <= 1 or len(np.unique(i[:, -1])) == 1:
                    print("Leaf node: Completely pure")
                    final_array.append(self.majority(data=i))
                    current_dict['class'] = self.majority(data=i) # Save as Leaf for test dict
                    continue

                # Guardrail 2: Max Depth Reached
                if n == self.depth - 1:
                    print("Leaf node: Max depth reached")
                    final_array.append(self.majority(data=i))
                    current_dict['class'] = self.majority(data=i) 
                    continue

                # Calculate the split!
                split_parent = self.node_split(i)
                left_data = split_parent[1]
                right_data = split_parent[2]

                # Guardrail 3: Split failed to divide data
                if len(left_data) == 0 or len(right_data) == 0:
                    print("Leaf node: Minimum features reached")
                    final_array.append(self.majority(data=i))
                    current_dict['class'] = self.majority(data=i) 
                    continue

                print(f"{'first' if n == 0 else 'left' if (x % 2) == 0 else 'right'} split: {split_parent[0][0]} <= {split_parent[0][1]}")
                results_array.append(self.majority(data= i))

                current_dict['feature'] = split_parent[0][0]
                current_dict['threshold'] = float(split_parent[0][1])
                current_dict['left'] = {}   # Create an empty dict for the left child
                current_dict['right'] = {}  # Create an empty dict for the right child
                
                # Append the newly created empty child dicts to the next layer
                next_dict_layer.append(current_dict['left'])
                next_dict_layer.append(current_dict['right'])

                next_layer.append(left_data)
                # print(f"{left_data.shape[0]} elements in left group")

                next_layer.append(right_data)
                # print(f"{right_data.shape[0]} elements in right group")


            data_matrix = next_layer
            dict_matrix = next_dict_layer
            n=n+1


        print(f"\nFinal majority classes: {final_array}, {len(final_array)} leaf nodes")
        print(self.tree_map)
        return(results_array)

    """
    Have now created a method to generate a decision tree to follow, however, we want to be able to test data against it.
    """
    

if __name__ == "__main__":
    run_model = decision_tree(data=train_set, observer=[0,1,2,3], predictor=4, max_depth= 10)
    run_model.decisiontree()

    prediction = run_model.predict(test_data)
    actual_truth = test_data[-1]
    
    print("\n--- TEST RESULTS ---")
    print(f"The model predicted: {prediction}")
    print(f"The actual flower was: {actual_truth}")
# %%
