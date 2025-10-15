import torch

# --- Creating Tensors ---

# From a Python list
my_list = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(my_list)
print("Tensor from list:")
print(tensor_from_list)

# A tensor of a specific shape with random values
random_tensor = torch.rand(3, 4) # 3 rows, 4 columns
print("\nA random tensor:")
print(random_tensor)

# --- Tensor Attributes ---
print("\nShape of the random tensor:")
print(random_tensor.shape)
print("Data type of the random tensor:")
print(random_tensor.dtype)

# --- Operations ---
# They work just like in NumPy
ones = torch.ones(3, 4)
result = random_tensor + ones
print("\nResult of addition:")
print(result)

#challenge 
zeros = torch.zeros(5,2)
print("\ntensor of zeros")
print(zeros)

tens_1 = torch.rand(3,4)
tens_2 = torch.rand(3,4)
product = tens_1 * tens_2
print("\n product")
print(tens_1)
print(tens_2)
print(product)
#returns element wise product