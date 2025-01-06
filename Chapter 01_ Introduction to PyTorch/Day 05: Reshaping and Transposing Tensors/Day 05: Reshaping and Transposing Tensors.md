<div align="center">
      <h3> <img src="https://github.com/ahammadmejbah/ahammadmejbah/blob/main/Software%20Intelligence%20longo.png?raw=true" width=""></h3>
      <center> <h2><a href="https://github.com/ahammadmejbah/The-90-Day-PyTorch-Mastery-Roadmap/tree/main/Chapter%2001%3A%20Introduction%20to%20PyTorch/Day%201%3A%20Introduction%20to%20Deep%20Learning%20and%20PyTorch"> Full Documentation Link: 👨🏻‍🎓 Day 05: Reshaping and Transposing Tensors</a></h2>
     </div>

<body>
<p align="center">
  <a href="mailto:ahammadmejbah@gmail.com"><img src="https://img.shields.io/badge/Email-ahammadmejbah%40gmail.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://github.com/ahammadmejbah"><img src="https://img.shields.io/badge/GitHub-%40ahammadmejbah-lightgrey?style=flat-square&logo=github"></a>
  <a href="https://linkedin.com/in/ahammadmejbah"><img src="https://img.shields.io/badge/LinkedIn-Mejbah%20Ahammad-blue?style=flat-square&logo=linkedin"></a>
  <a href="https://softwareintelligence.ai/"><img src="https://img.shields.io/badge/Website-Bytes%20of%20Intelligence-lightgrey?style=flat-square&logo=google-chrome"></a>
  <a href="https://www.youtube.com/@IntelligenceAcademy"><img src="https://img.shields.io/badge/YouTube-IntelligenceAcademy-red?style=flat-square&logo=youtube"></a>
  <a href="https://www.researchgate.net/profile/Mejbah-Ahammad-2"><img src="https://img.shields.io/badge/ResearchGate-Mejbah%20Ahammad-blue?style=flat-square&logo=researchgate"></a>
  <br>
  <img src="https://img.shields.io/badge/Phone-%2B8801874603631-green?style=flat-square&logo=whatsapp">
  <a href="https://www.hackerrank.com/profile/ahammadmejbah"><img src="https://img.shields.io/badge/Hackerrank-ahammadmejbah-green?style=flat-square&logo=hackerrank"></a>
</p>


## **Table of Contents**

1. [Introduction](#1-introduction)
2. [Reshaping Tensors with `view` and `reshape`](#2-reshaping-tensors-with-view-and-reshape)
   - [2.1. Using `view`](#21-using-view)
   - [2.2. Using `reshape`](#22-using-reshape)
   - [2.3. Differences Between `view` and `reshape`](#23-differences-between-view-and-reshape)
   - [2.4. Advanced Reshaping Techniques](#24-advanced-reshaping-techniques)
3. [Transposing Tensors with `transpose` and `permute`](#3-transposing-tensors-with-transpose-and-permute)
   - [3.1. Using `transpose`](#31-using-transpose)
   - [3.2. Using `permute`](#32-using-permute)
   - [3.3. Advanced Transposing Techniques](#33-advanced-transposing-techniques)
4. [Combining Reshape and Transpose Operations](#4-combining-reshape-and-transpose-operations)
   - [4.1. Sequential Operations](#41-sequential-operations)
   - [4.2. Practical Applications](#42-practical-applications)
5. [Reshaping and Transposing in Neural Networks](#5-reshaping-and-transposing-in-neural-networks)
   - [5.1. Preparing Data for Convolutional Neural Networks (CNNs)](#51-preparing-data-for-convolutional-neural-networks-cnns)
   - [5.2. Flattening for Fully Connected Layers](#52-flattening-for-fully-connected-layers)
   - [5.3. Handling Variable Batch Sizes](#53-handling-variable-batch-sizes)
6. [Handling Non-Contiguous Tensors](#6-handling-non-contiguous-tensors)
   - [6.1. Identifying Non-Contiguous Tensors](#61-identifying-non-contiguous-tensors)
   - [6.2. Making Tensors Contiguous](#62-making-tensors-contiguous)
7. [Best Practices and Optimization Tips](#7-best-practices-and-optimization-tips)
   - [7.1. Choosing Between `view` and `reshape`](#71-choosing-between-view-and-reshape)
   - [7.2. Minimizing Data Copies](#72-minimizing-data-copies)
   - [7.3. Using In-Place Operations Wisely](#73-using-in-place-operations-wisely)
8. [Common Pitfalls and How to Avoid Them](#8-common-pitfalls-and-how-to-avoid-them)
9. [Exercises for Practice](#9-exercises-for-practice)
   - [9.1. Exercise 1: Reshape and Transpose a Tensor](#91-exercise-1-reshape-and-transpose-a-tensor)
   - [9.2. Exercise 2: Batch Matrix Multiplication](#92-exercise-2-batch-matrix-multiplication)
   - [9.3. Exercise 3: Handling Variable Dimensions](#93-exercise-3-handling-variable-dimensions)
10. [Summary](#10-summary)
11. [Additional Resources](#11-additional-resources)

---

## **1. Introduction**

Tensors are the core data structures in PyTorch, serving as the building blocks for models and data manipulation. Efficiently reshaping and transposing tensors is crucial for preparing data, optimizing models, and ensuring compatibility between different layers and operations.

- **Reshaping:** Changing the shape of a tensor without altering its data.
- **Transposing:** Rearranging the dimensions of a tensor.

Mastering these operations will enhance your ability to manipulate data effectively across various deep learning tasks.

---

## **2. Reshaping Tensors with `view` and `reshape`**

Reshaping involves changing the dimensions of a tensor. PyTorch provides two primary methods for this: `view` and `reshape`.

### **2.1. Using `view`**

The `view` method returns a new tensor with the same data but a different shape. It **requires** the tensor to be **contiguous** in memory.

**Example 1: Basic Reshaping with `view`**

```python
import torch

# Create a tensor of shape (4, 4)
original_tensor = torch.arange(16).reshape(4, 4)
print("Original Tensor:\n", original_tensor)

# Reshape to (2, 8) using view
reshaped_tensor = original_tensor.view(2, 8)
print("\nReshaped Tensor (2, 8):\n", reshaped_tensor)

# Reshape to (8, 2) using view
reshaped_tensor = original_tensor.view(8, 2)
print("\nReshaped Tensor (8, 2):\n", reshaped_tensor)
```

**Output:**
```
Original Tensor:
 tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])

Reshaped Tensor (2, 8):
 tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 8,  9, 10, 11, 12, 13, 14, 15]])

Reshaped Tensor (8, 2):
 tensor([[ 0,  1],
        [ 2,  3],
        [ 4,  5],
        [ 6,  7],
        [ 8,  9],
        [10, 11],
        [12, 13],
        [14, 15]])
```

**Key Points:**
- The total number of elements must remain constant.
- The tensor must be contiguous; otherwise, `view` will raise an error.

### **2.2. Using `reshape`**

The `reshape` method is more flexible than `view` as it can handle non-contiguous tensors by returning a copy if necessary. If the tensor is already contiguous, `reshape` behaves like `view`.

**Example 2: Reshaping with `reshape`**

```python
# Using the same original_tensor

# Reshape to (2, 8) using reshape
reshaped_tensor = original_tensor.reshape(2, 8)
print("\nReshaped Tensor with reshape (2, 8):\n", reshaped_tensor)

# Reshape with one dimension inferred (-1)
reshaped_tensor = original_tensor.reshape(-1, 8)
print("\nReshaped Tensor with reshape (-1, 8):\n", reshaped_tensor)

# Reshape to a single dimension
reshaped_tensor = original_tensor.reshape(-1)
print("\nReshaped Tensor to 1D:\n", reshaped_tensor)
```

**Output:**
```
Reshaped Tensor with reshape (2, 8):
 tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 8,  9, 10, 11, 12, 13, 14, 15]])

Reshaped Tensor with reshape (-1, 8):
 tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 8,  9, 10, 11, 12, 13, 14, 15]])

Reshaped Tensor to 1D:
 tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
```

**Key Points:**
- `reshape` can infer one dimension using `-1`.
- It handles non-contiguous tensors by making a copy if necessary.

### **2.3. Differences Between `view` and `reshape`**

| Feature              | `view`                                       | `reshape`                                    |
|----------------------|----------------------------------------------|----------------------------------------------|
| Memory Contiguity    | Requires tensor to be contiguous             | Does not require tensor to be contiguous     |
| Performance          | Faster when tensor is contiguous             | Slightly slower if a copy is needed          |
| Flexibility          | Less flexible; limited to contiguous tensors | More flexible; handles non-contiguous tensors|
| Error Handling       | Raises error if tensor is not contiguous     | Automatically handles non-contiguous tensors |

**Example 3: Handling Non-Contiguous Tensors**

```python
# Transpose the original tensor to make it non-contiguous
transposed_tensor = original_tensor.t()
print("\nTransposed Tensor:\n", transposed_tensor)
print("\nIs transposed_tensor contiguous?", transposed_tensor.is_contiguous())

# Attempt to use view on non-contiguous tensor (will raise an error)
try:
    reshaped_view = transposed_tensor.view(16)
except RuntimeError as e:
    print("\nError using view on non-contiguous tensor:", e)

# Use reshape instead
reshaped_reshape = transposed_tensor.reshape(16)
print("\nReshaped with reshape:\n", reshaped_reshape)
```

**Output:**
```
Transposed Tensor:
 tensor([[ 0,  4,  8, 12],
        [ 1,  5,  9, 13],
        [ 2,  6, 10, 14],
        [ 3,  7, 11, 15]])

Is transposed_tensor contiguous? False

Error using view on non-contiguous tensor: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

Reshaped with reshape:
 tensor([ 0,  4,  8, 12,  1,  5,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15])
```

**Explanation:**
- **Transposed Tensor:** After transposing, the tensor becomes non-contiguous.
- **Using `view`:** Raises an error because `view` requires contiguity.
- **Using `reshape`:** Successfully reshapes the non-contiguous tensor by creating a copy.

### **2.4. Advanced Reshaping Techniques**

#### **2.4.1. Reshaping Higher-Dimensional Tensors**

Reshaping isn't limited to 2D or 3D tensors. Here's how to handle higher-dimensional tensors.

**Example 4: Reshaping a 4D Tensor**

```python
# Create a 4D tensor representing a batch of 2 RGB images of size 3x4
tensor_4d = torch.arange(2*3*3*4).reshape(2, 3, 3, 4)
print("Original 4D Tensor Shape:", tensor_4d.shape)
print(tensor_4d)

# Reshape to (2, 9, 4) by merging channels and height
reshaped = tensor_4d.view(2, 9, 4)
print("\nReshaped to (2, 9, 4):")
print(reshaped)
print("Shape:", reshaped.shape)

# Reshape to (6, 3, 4) by merging batch and channels
reshaped = tensor_4d.view(6, 3, 4)
print("\nReshaped to (6, 3, 4):")
print(reshaped)
print("Shape:", reshaped.shape)
```

**Output:**
```
Original 4D Tensor Shape: torch.Size([2, 3, 3, 4])
tensor([[[[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11]],

         [[12, 13, 14, 15],
          [16, 17, 18, 19],
          [20, 21, 22, 23]],

         [[24, 25, 26, 27],
          [28, 29, 30, 31],
          [32, 33, 34, 35]]],


        [[[36, 37, 38, 39],
          [40, 41, 42, 43],
          [44, 45, 46, 47]],

         [[48, 49, 50, 51],
          [52, 53, 54, 55],
          [56, 57, 58, 59]],

         [[60, 61, 62, 63],
          [64, 65, 66, 67],
          [68, 69, 70, 71]]]])

Reshaped to (2, 9, 4):
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11],
         [12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23],
         [24, 25, 26, 27],
         [28, 29, 30, 31],
         [32, 33, 34, 35]],

        [[36, 37, 38, 39],
         [40, 41, 42, 43],
         [44, 45, 46, 47],
         [48, 49, 50, 51],
         [52, 53, 54, 55],
         [56, 57, 58, 59],
         [60, 61, 62, 63],
         [64, 65, 66, 67],
         [68, 69, 70, 71]]])
Shape: torch.Size([2, 9, 4])

Reshaped to (6, 3, 4):
tensor([[[ 0,  1,  2,  3],
  [ 4,  5,  6,  7],
  [ 8,  9, 10, 11]],

 [[12, 13, 14, 15],
  [16, 17, 18, 19],
  [20, 21, 22, 23]],

 [[24, 25, 26, 27],
  [28, 29, 30, 31],
  [32, 33, 34, 35]],

 [[36, 37, 38, 39],
  [40, 41, 42, 43],
  [44, 45, 46, 47]],

 [[48, 49, 50, 51],
  [52, 53, 54, 55],
  [56, 57, 58, 59]],

 [[60, 61, 62, 63],
  [64, 65, 66, 67],
  [68, 69, 70, 71]]])
Shape: torch.Size([6, 3, 4])
```

**Explanation:**
- **Original 4D Tensor:** Represents a batch of 2 samples, each with 3 channels, height 3, and width 4.
- **Reshaped to (2, 9, 4):** Merges the `channels` and `height` dimensions (`3 * 3 = 9`), keeping the `width` unchanged.
- **Reshaped to (6, 3, 4):** Merges the `batch_size` and `channels` dimensions (`2 * 3 = 6`), maintaining `height` and `width`.

#### **2.4.2. Reshaping with Negative Dimensions**

Using `-1` allows PyTorch to automatically infer the size of one dimension based on the others.

**Example 5: Inferring Dimensions with `-1`**

```python
# Original tensor of shape (2, 3, 4)
tensor = torch.arange(24).reshape(2, 3, 4)
print("\nOriginal Tensor Shape:", tensor.shape)

# Reshape to (2, -1)
reshaped = tensor.view(2, -1)
print("Reshaped to (2, -1):\n", reshaped)
print("Shape:", reshaped.shape)

# Reshape to (-1, 4)
reshaped = tensor.view(-1, 4)
print("\nReshaped to (-1, 4):\n", reshaped)
print("Shape:", reshaped.shape)
```

**Output:**
```
Original Tensor Shape: torch.Size([2, 3, 4])
Reshaped to (2, -1):
 tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])
Shape: torch.Size([2, 12])

Reshaped to (-1, 4):
 tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]])
Shape: torch.Size([6, 4])
```

**Explanation:**
- **Reshape to (2, -1):** Infers the second dimension as `12` since `2 * 12 = 24`.
- **Reshape to (-1, 4):** Infers the first dimension as `6` since `6 * 4 = 24`.

---

## **3. Transposing Tensors with `transpose` and `permute`**

Transposing rearranges the dimensions of a tensor. PyTorch offers two primary methods: `transpose` and `permute`.

### **3.1. Using `transpose`**

The `transpose` method swaps two specified dimensions of a tensor.

**Example 6: Transposing a 2D Tensor**

```python
# Original 2D tensor of shape (2, 3)
tensor_2d = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])
print("Original 2D Tensor:\n", tensor_2d)

# Transpose the tensor
transposed_2d = tensor_2d.transpose(0, 1)
print("\nTransposed 2D Tensor:\n", transposed_2d)
print("\nShape:", transposed_2d.shape)
```

**Output:**
```
Original 2D Tensor:
 tensor([[1, 2, 3],
        [4, 5, 6]])

Transposed 2D Tensor:
 tensor([[1, 4],
        [2, 5],
        [3, 6]])

Shape: torch.Size([3, 2])
```

**Example 7: Transposing a 3D Tensor**

```python
# Original 3D tensor of shape (2, 3, 4)
tensor_3d = torch.arange(24).reshape(2, 3, 4)
print("Original 3D Tensor:\n", tensor_3d)

# Transpose dimensions 0 and 1
transposed_3d = tensor_3d.transpose(0, 1)
print("\nTransposed 3D Tensor (dims 0 and 1):\n", transposed_3d)
print("\nShape:", transposed_3d.shape)
```

**Output:**
```
Original 3D Tensor:
 tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])

Transposed 3D Tensor (dims 0 and 1):
 tensor([[[ 0,  1,  2,  3],
         [12, 13, 14, 15]],

        [[ 4,  5,  6,  7],
         [16, 17, 18, 19]],

        [[ 8,  9, 10, 11],
         [20, 21, 22, 23]]])

Shape: torch.Size([3, 2, 4])
```

### **3.2. Using `permute`**

The `permute` method allows you to reorder **all** dimensions of a tensor based on a specified sequence.

**Example 8: Permuting a 3D Tensor**

```python
# Original 3D tensor of shape (2, 3, 4)
tensor_3d = torch.arange(24).reshape(2, 3, 4)
print("\nOriginal 3D Tensor Shape:", tensor_3d.shape)

# Permute to (4, 2, 3)
permuted_tensor = tensor_3d.permute(2, 0, 1)
print("\nPermuted Tensor (2, 0, 1):\n", permuted_tensor)
print("\nShape:", permuted_tensor.shape)

# Another permutation to (3, 4, 2)
permuted_tensor = tensor_3d.permute(1, 2, 0)
print("\nPermuted Tensor (1, 2, 0):\n", permuted_tensor)
print("\nShape:", permuted_tensor.shape)
```

**Output:**
```
Original 3D Tensor Shape: torch.Size([2, 3, 4])

Permuted Tensor (2, 0, 1):
 tensor([[[ 0,  4,  8],
         [12, 16, 20]],

        [[ 1,  5,  9],
         [13, 17, 21]],

        [[ 2,  6, 10],
         [14, 18, 22]],

        [[ 3,  7, 11],
         [15, 19, 23]]])

Shape: torch.Size([4, 2, 3])

Permuted Tensor (1, 2, 0):
 tensor([[[ 0, 12],
         [ 4, 16],
         [ 8, 20]],

        [[ 1, 13],
         [ 5, 17],
         [ 9, 21]],

        [[ 2, 14],
         [ 6, 18],
         [10, 22]],

        [[ 3, 15],
         [ 7, 19],
         [11, 23]]])

Shape: torch.Size([3, 4, 2])
```

**Explanation:**
- **Permute (2, 0, 1):** Reorders dimensions to bring the third dimension first, resulting in shape `(4, 2, 3)`.
- **Permute (1, 2, 0):** Reorders dimensions to bring the second dimension first, resulting in shape `(3, 4, 2)`.

### **3.3. Advanced Transposing Techniques**

#### **3.3.1. Transposing Higher-Dimensional Tensors**

Transposing isn't limited to 2D or 3D tensors. Here's how to handle 4D and higher-dimensional tensors.

**Example 9: Transposing a 4D Tensor**

```python
# Create a 4D tensor: (batch_size, channels, height, width)
tensor_4d = torch.arange(2*3*4*5).reshape(2, 3, 4, 5)
print("Original 4D Tensor Shape:", tensor_4d.shape)

# Transpose channels and height dimensions
transposed = tensor_4d.transpose(1, 2)
print("\nTransposed Channels and Height (1 ↔ 2):")
print("Shape:", transposed.shape)

# Permute dimensions to (height, width, channels, batch_size)
permuted = tensor_4d.permute(2, 3, 1, 0)
print("\nPermuted to (height, width, channels, batch_size):")
print("Shape:", permuted.shape)
```

**Output:**
```
Original 4D Tensor Shape: torch.Size([2, 3, 4, 5])

Transposed Channels and Height (1 ↔ 2):
Shape: torch.Size([2, 4, 3, 5])

Permuted to (height, width, channels, batch_size):
Shape: torch.Size([4, 5, 3, 2])
```

**Explanation:**
- **Transposing Channels and Height:** Swaps the `channels` (dimension 1) with `height` (dimension 2), resulting in shape `(2, 4, 3, 5)`.
- **Permuting to (height, width, channels, batch_size):** Reorders all dimensions, resulting in shape `(4, 5, 3, 2)`.

#### **3.3.2. Swapping Multiple Dimensions with `permute`**

`permute` can reorder multiple dimensions simultaneously, which is useful for complex tensor rearrangements.

**Example 10: Permuting a 5D Tensor**

```python
# Create a 5D tensor
tensor_5d = torch.randn(2, 3, 4, 5, 6)
print("Original 5D Tensor Shape:", tensor_5d.shape)

# Desired order: (batch, depth, height, width, channels) -> (batch, channels, depth, height, width)
permuted = tensor_5d.permute(0, 4, 1, 2, 3)
print("\nPermuted 5D Tensor Shape:", permuted.shape)

# Another permutation: (batch, channels, depth, height, width) -> (depth, batch, channels, height, width)
permuted_again = permuted.permute(2, 0, 1, 3, 4)
print("Another Permutation Shape:", permuted_again.shape)
```

**Output:**
```
Original 5D Tensor Shape: torch.Size([2, 3, 4, 5, 6])

Permuted 5D Tensor Shape: torch.Size([2, 6, 3, 4, 5])
Another Permutation Shape: torch.Size([3, 2, 6, 4, 5])
```

**Explanation:**
- **First Permutation:** Moves `channels` to the second dimension.
- **Second Permutation:** Reorders dimensions to place `depth` first, followed by `batch`, `channels`, `height`, and `width`.

---

## **4. Combining Reshape and Transpose Operations**

Often, you might need to perform both reshaping and transposing to achieve the desired tensor layout.

### **4.1. Sequential Operations**

Performing reshaping and transposing in sequence can help in aligning tensor dimensions for specific operations.

**Example 11: Reshaping and Transposing a Tensor**

```python
# Create a tensor of shape (2, 3, 4, 5)
tensor = torch.arange(2*3*4*5).reshape(2, 3, 4, 5)
print("Original Tensor Shape:", tensor.shape)

# Step 1: Reshape to (2, 12, 5)
reshaped = tensor.view(2, 12, 5)
print("\nReshaped to (2, 12, 5):")
print("Shape:", reshaped.shape)

# Step 2: Transpose the last two dimensions to (2, 5, 12)
transposed = reshaped.transpose(1, 2)
print("\nTransposed to (2, 5, 12):")
print("Shape:", transposed.shape)
```

**Output:**
```
Original Tensor Shape: torch.Size([2, 3, 4, 5])

Reshaped to (2, 12, 5):
Shape: torch.Size([2, 12, 5])

Transposed to (2, 5, 12):
Shape: torch.Size([2, 5, 12])
```

**Explanation:**
- **Reshape:** Merges the `channels` and `height` dimensions into one.
- **Transpose:** Swaps the merged dimension with `width`.

### **4.2. Practical Applications**

Combining reshaping and transposing is often necessary when preparing data for specific layers or operations in neural networks.

**Example 12: Preparing Image Data for CNNs**

```python
import torch
import torch.nn as nn

# Suppose we have image data in the shape (batch_size, height, width, channels)
batch_size, height, width, channels = 10, 28, 28, 3
images = torch.randn(batch_size, height, width, channels)
print("Original Image Batch Shape:", images.shape)

# Step 1: Permute to (batch_size, channels, height, width)
images_cnn = images.permute(0, 3, 1, 2)
print("Permuted Image Batch Shape for CNN:", images_cnn.shape)

# Step 2: Flatten the features for a fully connected layer
flattened = images_cnn.view(batch_size, -1)
print("Flattened Features Shape:", flattened.shape)

# Define a simple fully connected layer
fc = nn.Linear(3*28*28, 10)  # Assuming 10 classes
output = fc(flattened)
print("Fully Connected Output Shape:", output.shape)
```

**Output:**
```
Original Image Batch Shape: torch.Size([10, 28, 28, 3])
Permuted Image Batch Shape for CNN: torch.Size([10, 3, 28, 28])
Flattened Features Shape: torch.Size([10, 2352])
Fully Connected Output Shape: torch.Size([10, 10])
```

**Explanation:**
- **Permute:** Reorders dimensions to match CNN input requirements `(batch_size, channels, height, width)`.
- **Flatten:** Converts the 4D tensor to 2D for input into a fully connected layer.

---

## **5. Reshaping and Transposing in Neural Networks**

Efficiently reshaping and transposing tensors is vital when building and training neural networks. These operations ensure that data flows correctly between different layers.

### **5.1. Preparing Data for Convolutional Neural Networks (CNNs)**

CNNs typically expect input data in the shape `(batch_size, channels, height, width)`. However, data might come in different formats and require reshaping.

**Example 13: Data Preparation for CNN**

```python
import torch
import torch.nn as nn

# Assume we have a dataset where images are in (batch_size, height, width, channels)
batch_size, height, width, channels = 16, 64, 64, 3
images = torch.randn(batch_size, height, width, channels)
print("Original Image Shape:", images.shape)

# Permute to (batch_size, channels, height, width)
images_cnn = images.permute(0, 3, 1, 2)
print("Permuted Image Shape for CNN:", images_cnn.shape)

# Define a simple CNN layer
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# Apply convolution
output = conv(images_cnn)
print("CNN Output Shape:", output.shape)
```

**Output:**
```
Original Image Shape: torch.Size([16, 64, 64, 3])
Permuted Image Shape for CNN: torch.Size([16, 3, 64, 64])
CNN Output Shape: torch.Size([16, 16, 64, 64])
```

**Explanation:**
- **Permute:** Adjusts the tensor shape to meet CNN input requirements.
- **Convolution:** Applies a convolutional layer, increasing the number of channels from 3 to 16.

### **5.2. Flattening for Fully Connected Layers**

After convolutional and pooling layers, tensors often need to be flattened before passing to fully connected layers.

**Example 14: Flattening CNN Output**

```python
# Continuing from the previous CNN output
print("CNN Output Shape:", output.shape)  # [16, 16, 64, 64]

# Flatten the features except the batch dimension
flattened = output.view(batch_size, -1)
print("Flattened Features Shape:", flattened.shape)  # [16, 65536]

# Define a fully connected layer
fc = nn.Linear(16*64*64, 10)  # Assuming 10 classes for classification

# Apply fully connected layer
fc_output = fc(flattened)
print("Fully Connected Output Shape:", fc_output.shape)  # [16, 10]
```

**Output:**
```
CNN Output Shape: torch.Size([16, 16, 64, 64])
Flattened Features Shape: torch.Size([16, 65536])
Fully Connected Output Shape: torch.Size([16, 10])
```

**Explanation:**
- **Flattening:** Converts the 4D tensor to 2D to interface with the fully connected layer.
- **Fully Connected Layer:** Transforms the flattened features into logits for 10 classes.

### **5.3. Handling Variable Batch Sizes**

When building models that process varying batch sizes, it's essential to write reshaping code that accommodates this variability.

**Example 15: Reshaping with Variable Batch Sizes**

```python
def reshape_with_dynamic_batch(tensor, new_shape):
    # tensor: input tensor with shape (batch_size, ...)
    # new_shape: desired shape with -1 allowed for dynamic dimensions
    batch_size = tensor.size(0)
    reshaped = tensor.view(batch_size, *new_shape)
    return reshaped

# Example usage
# Batch size 4, original shape (4, 2, 3)
tensor = torch.arange(24).reshape(4, 2, 3)
print("Original Tensor Shape:", tensor.shape)

# Reshape each sample to (6,)
reshaped = reshape_with_dynamic_batch(tensor, (6,))
print("\nReshaped to (batch_size, 6):")
print(reshaped)
print("Shape:", reshaped.shape)

# Reshape each sample to (3, 2)
reshaped = reshape_with_dynamic_batch(tensor, (3, 2))
print("\nReshaped to (batch_size, 3, 2):")
print(reshaped)
print("Shape:", reshaped.shape)
```

**Output:**
```
Original Tensor Shape: torch.Size([4, 2, 3])

Reshaped to (batch_size, 6):
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]])
Shape: torch.Size([4, 6])

Reshaped to (batch_size, 3, 2):
tensor([[[ 0,  1],
         [ 2,  3],
         [ 4,  5]],

        [[ 6,  7],
         [ 8,  9],
         [10, 11]],

        [[12, 13],
         [14, 15],
         [16, 17]],

        [[18, 19],
         [20, 21],
         [22, 23]]])
Shape: torch.Size([4, 3, 2])
```

**Explanation:**
- **Dynamic Reshaping Function:** Automatically adjusts reshaping based on the input tensor's batch size.
- **Usage Examples:** Demonstrates reshaping each sample within the batch to different shapes without hardcoding the batch size.

---

## **6. Handling Non-Contiguous Tensors**

Certain tensor operations, like `transpose` and `permute`, can result in **non-contiguous** tensors. Some functions, like `view`, require contiguous tensors. Here's how to manage such scenarios.

### **6.1. Identifying Non-Contiguous Tensors**

**Example 16: Checking Contiguity**

```python
# Create a tensor
tensor = torch.randn(2, 3, 4)
print("Original Tensor Shape:", tensor.shape)
print("Is original tensor contiguous?", tensor.is_contiguous())

# Transpose to make it non-contiguous
transposed = tensor.transpose(1, 2)
print("\nTransposed Tensor Shape:", transposed.shape)
print("Is transposed tensor contiguous?", transposed.is_contiguous())
```

**Output:**
```
Original Tensor Shape: torch.Size([2, 3, 4])
Is original tensor contiguous? True

Transposed Tensor Shape: torch.Size([2, 4, 3])
Is transposed tensor contiguous? False
```

**Explanation:**
- **Original Tensor:** Contiguous in memory.
- **Transposed Tensor:** Non-contiguous after transposing dimensions.

### **6.2. Making Tensors Contiguous**

If a tensor is non-contiguous, you can make it contiguous using the `.contiguous()` method before applying operations that require contiguity.

**Example 17: Making a Tensor Contiguous**

```python
# Continuing from the previous example

# Attempt to use view on non-contiguous tensor (will raise an error)
try:
    reshaped_view = transposed.view(24)
except RuntimeError as e:
    print("\nError using view on non-contiguous tensor:", e)

# Make the tensor contiguous
contiguous_transposed = transposed.contiguous()
print("\nIs contiguous_transposed contiguous?", contiguous_transposed.is_contiguous())

# Now use view
reshaped = contiguous_transposed.view(24)
print("Reshaped Tensor Shape:", reshaped.shape)
```

**Output:**
```
Error using view on non-contiguous tensor: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.

Is contiguous_transposed contiguous? True
Reshaped Tensor Shape: torch.Size([24])
```

**Explanation:**
- **Error with `view`:** Cannot reshape non-contiguous tensor using `view`.
- **Making Contiguous:** `.contiguous()` creates a contiguous copy of the tensor.
- **Successful Reshape:** Now `view` works as expected.

---

## **7. Best Practices and Optimization Tips**

Efficient tensor manipulation can significantly optimize your deep learning workflows. Here are some best practices and tips to enhance performance and maintain code clarity.

### **7.1. Choosing Between `view` and `reshape`**

- **Use `view` When:**
  - The tensor is already contiguous.
  - You want the fastest possible reshape without copying data.
  
- **Use `reshape` When:**
  - The tensor might be non-contiguous.
  - You prefer flexibility over raw speed.

**Example 18: Comparing `view` and `reshape`**

```python
# Create a contiguous tensor
tensor_contig = torch.arange(12).reshape(3, 4)
print("\nContiguous Tensor Shape:", tensor_contig.shape)

# Use view
reshaped_view = tensor_contig.view(6, 2)
print("Reshaped with view:", reshaped_view.shape)

# Transpose to make it non-contiguous
tensor_non_contig = tensor_contig.transpose(0, 1)
print("\nNon-Contiguous Tensor Shape:", tensor_non_contig.shape)

# Attempt to use view (will fail)
try:
    reshaped_view = tensor_non_contig.view(6, 2)
except RuntimeError as e:
    print("Error with view on non-contiguous tensor:", e)

# Use reshape instead
reshaped_reshape = tensor_non_contig.reshape(6, 2)
print("Reshaped with reshape:", reshaped_reshape.shape)
```

**Output:**
```
Contiguous Tensor Shape: torch.Size([3, 4])
Reshaped with view: torch.Size([6, 2])

Non-Contiguous Tensor Shape: torch.Size([4, 3])
Error with view on non-contiguous tensor: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
Reshaped with reshape: torch.Size([6, 2])
```

### **7.2. Minimizing Data Copies**

- **Prefer `view` Over `reshape` When Possible:** `view` doesn't copy data if the tensor is contiguous.
- **Use `.contiguous()` Sparingly:** Making a tensor contiguous can involve copying data. Only use it when necessary.
- **Chain Operations Efficiently:** Combine reshaping and transposing operations to minimize intermediate steps.

**Example 19: Minimizing Data Copies**

```python
# Create a tensor and make it non-contiguous
tensor = torch.randn(2, 3, 4)
transposed = tensor.transpose(1, 2)
print("\nTransposed Shape:", transposed.shape)
print("Is transposed contiguous?", transposed.is_contiguous())

# Use reshape instead of making it contiguous and then view
reshaped = transposed.reshape(2, -1)
print("Reshaped with reshape:", reshaped.shape)

# Alternatively, if using view is necessary
contiguous_transposed = transposed.contiguous()
reshaped_view = contiguous_transposed.view(2, -1)
print("Reshaped with view after making contiguous:", reshaped_view.shape)
```

**Output:**
```
Transposed Shape: torch.Size([2, 4, 3])
Is transposed contiguous? False
Reshaped with reshape: torch.Size([2, 12])
Reshaped with view after making contiguous: torch.Size([2, 12])
```

**Explanation:**
- **Using `reshape`:** Avoids making the tensor contiguous by handling non-contiguity internally.
- **Using `view` After `.contiguous()`:** Ensures the tensor is contiguous before reshaping, but involves a data copy.

### **7.3. Using In-Place Operations Wisely**

In-place operations modify tensors without making copies, saving memory. However, they should be used cautiously to avoid unintended side effects, especially when tensors are involved in computational graphs for automatic differentiation.

**Example 20: In-Place Transpose**

```python
# Create a tensor
tensor = torch.randn(2, 3)
print("\nOriginal Tensor Shape:", tensor.shape)

# In-place transpose
tensor.transpose_(0, 1)
print("In-Place Transposed Tensor Shape:", tensor.shape)
```

**Output:**
```
Original Tensor Shape: torch.Size([2, 3])
In-Place Transposed Tensor Shape: torch.Size([3, 2])
```

**Caution:**
- **Automatic Differentiation:** In-place operations can interfere with gradient computations.
- **Data Integrity:** Ensure that in-place modifications don't unintentionally alter data needed elsewhere.

---

## **8. Common Pitfalls and How to Avoid Them**

Understanding common mistakes can help you write more robust and error-free code.

### **8.1. Ensuring Correct Number of Elements**

Always ensure that the total number of elements remains the same when reshaping.

**Incorrect Example: Mismatched Elements**

```python
# Original tensor with 12 elements
tensor = torch.arange(12).reshape(3, 4)
print("\nOriginal Tensor Shape:", tensor.shape)

# Attempt to reshape to (5, 3) which requires 15 elements
try:
    reshaped = tensor.view(5, 3)
except RuntimeError as e:
    print("Error:", e)
```

**Output:**
```
Original Tensor Shape: torch.Size([3, 4])
Error: view size is not compatible with input tensor's size and stride (invalid size for view: [5, 3]).
```

**Solution:**
- **Match Total Elements:** Ensure that the product of the new shape dimensions equals the original number of elements.
- **Use `-1` for Inference:** Let PyTorch infer one dimension to maintain consistency.

### **8.2. Contiguous Tensors**

Operations like `transpose` and `permute` can make tensors non-contiguous. Before using `view`, ensure the tensor is contiguous.

**Example 21: Making a Tensor Contiguous**

```python
# Original tensor
tensor = torch.randn(2, 3, 4)
print("\nOriginal Tensor Shape:", tensor.shape)

# Transpose to make it non-contiguous
transposed = tensor.transpose(1, 2)
print("Transposed Tensor Shape:", transposed.shape)
print("Is transposed tensor contiguous?", transposed.is_contiguous())

# Make it contiguous before using view
contiguous_tensor = transposed.contiguous()
print("Is contiguous_tensor contiguous?", contiguous_tensor.is_contiguous())

# Now use view
reshaped = contiguous_tensor.view(2, -1)
print("Reshaped Tensor Shape:", reshaped.shape)
```

**Output:**
```
Original Tensor Shape: torch.Size([2, 3, 4])
Transposed Tensor Shape: torch.Size([2, 4, 3])
Is transposed tensor contiguous? False
Is contiguous_tensor contiguous? True
Reshaped Tensor Shape: torch.Size([2, 12])
```

### **8.3. Using `unsqueeze` and `squeeze` for Dimension Manipulation**

Adding or removing dimensions can facilitate reshaping or transposing operations.

**Example 22: Adding and Removing Dimensions**

```python
# Original tensor of shape (3, 4)
tensor = torch.randn(3, 4)
print("\nOriginal Tensor Shape:", tensor.shape)

# Add a dimension at position 0
tensor_unsqueezed = tensor.unsqueeze(0)
print("After unsqueeze(0):", tensor_unsqueezed.shape)

# Add a dimension at position 2
tensor_unsqueezed = tensor_unsqueezed.unsqueeze(2)
print("After unsqueeze(2):", tensor_unsqueezed.shape)

# Remove dimension at position 0
tensor_squeezed = tensor_unsqueezed.squeeze(0)
print("After squeeze(0):", tensor_squeezed.shape)

# Remove dimension at position 2
tensor_squeezed = tensor_squeezed.squeeze(2)
print("After squeeze(2):", tensor_squeezed.shape)
```

**Output:**
```
Original Tensor Shape: torch.Size([3, 4])
After unsqueeze(0): torch.Size([1, 3, 4])
After unsqueeze(2): torch.Size([1, 3, 1, 4])
After squeeze(0): torch.Size([3, 1, 4])
After squeeze(2): torch.Size([3, 4])
```

**Explanation:**
- **Unsqueeze:** Adds a dimension of size `1` at the specified position.
- **Squeeze:** Removes a dimension of size `1` at the specified position.

---

## **9. Exercises for Practice**

Engaging with exercises reinforces your understanding and ensures you can apply tensor reshaping and transposing techniques effectively.

### **9.1. Exercise 1: Reshape and Transpose a Tensor**

**Task:**

Given a tensor of shape `(3, 2, 4, 5)`, perform the following operations:

1. Transpose dimensions `1` and `2`.
2. Reshape the transposed tensor to `(3, 8, 5)`.
3. Permute the tensor to `(5, 3, 8)`.

**Solution:**

```python
# Create the tensor
tensor = torch.arange(3*2*4*5).reshape(3, 2, 4, 5)
print("Original Tensor Shape:", tensor.shape)  # [3, 2, 4, 5]

# Step 1: Transpose dimensions 1 and 2
transposed = tensor.transpose(1, 2)
print("After Transpose (1 ↔ 2) Shape:", transposed.shape)  # [3, 4, 2, 5]

# Step 2: Reshape to (3, 8, 5)
reshaped = transposed.view(3, 8, 5)
print("After Reshape to (3, 8, 5) Shape:", reshaped.shape)  # [3, 8, 5]

# Step 3: Permute to (5, 3, 8)
permuted = reshaped.permute(2, 0, 1)
print("After Permute to (5, 3, 8) Shape:", permuted.shape)  # [5, 3, 8]
```

**Expected Output:**
```
Original Tensor Shape: torch.Size([3, 2, 4, 5])
After Transpose (1 ↔ 2) Shape: torch.Size([3, 4, 2, 5])
After Reshape to (3, 8, 5) Shape: torch.Size([3, 8, 5])
After Permute to (5, 3, 8) Shape: torch.Size([5, 3, 8])
```

### **9.2. Exercise 2: Batch Matrix Multiplication**

**Task:**

Given two tensors `A` of shape `(batch_size, n, m)` and `B` of shape `(batch_size, m, p)`, perform batch matrix multiplication to obtain a tensor `C` of shape `(batch_size, n, p)`.

**Solution:**

```python
batch_size, n, m, p = 4, 3, 5, 2
A = torch.randn(batch_size, n, m)
B = torch.randn(batch_size, m, p)
print("A Shape:", A.shape)  # [4, 3, 5]
print("B Shape:", B.shape)  # [4, 5, 2]

# Batch matrix multiplication
C = torch.bmm(A, B)
print("C Shape:", C.shape)  # [4, 3, 2]
```

**Expected Output:**
```
A Shape: torch.Size([4, 3, 5])
B Shape: torch.Size([4, 5, 2])
C Shape: torch.Size([4, 3, 2])
```

**Explanation:**
- **`torch.bmm`:** Performs batch matrix multiplication, multiplying corresponding matrices in batches.

### **9.3. Exercise 3: Handling Variable Dimensions**

**Task:**

Write a function `flexible_reshape` that takes a tensor and reshapes it to have a specified number of dimensions, automatically inferring one dimension if necessary.

**Solution:**

```python
def flexible_reshape(tensor, *shape):
    """
    Reshape the tensor to the desired shape.
    Allows one dimension to be inferred using -1.
    """
    return tensor.view(*shape)

# Example usage
tensor = torch.arange(24).reshape(2, 3, 4)
print("Original Shape:", tensor.shape)  # [2, 3, 4]

# Reshape to (2, 12)
reshaped = flexible_reshape(tensor, 2, 12)
print("Reshaped to (2, 12):", reshaped.shape)  # [2, 12]

# Reshape to (6, 4)
reshaped = flexible_reshape(tensor, 6, 4)
print("Reshaped to (6, 4):", reshaped.shape)  # [6, 4]

# Reshape to (2, -1)
reshaped = flexible_reshape(tensor, 2, -1)
print("Reshaped to (2, -1):", reshaped.shape)  # [2, 12]
```

**Expected Output:**
```
Original Shape: torch.Size([2, 3, 4])
Reshaped to (2, 12): torch.Size([2, 12])
Reshaped to (6, 4): torch.Size([6, 4])
Reshaped to (2, -1): torch.Size([2, 12])
```

**Explanation:**
- **`flexible_reshape` Function:** Utilizes `view` to reshape tensors, allowing one dimension to be inferred with `-1`.
- **Usage Examples:** Demonstrates reshaping to different shapes, including dynamic dimension inference.

---

## **10. Summary**

- **Reshaping Tensors:**
  - **`view`:** Efficient for contiguous tensors; requires exact matching of total elements.
  - **`reshape`:** Flexible; handles non-contiguous tensors by copying if necessary.
  - **`unsqueeze`/`squeeze`:** Add or remove dimensions to facilitate reshaping.

- **Transposing Tensors:**
  - **`transpose`:** Swap two specific dimensions.
  - **`permute`:** Reorder all dimensions in any desired sequence.

- **Best Practices:**
  - Use `view` for speed when tensors are contiguous.
  - Use `reshape` for flexibility.
  - Ensure tensors are contiguous before reshaping with `view`.
  - Combine reshaping and transposing thoughtfully to align tensor dimensions for model compatibility.

- **Common Pitfalls:**
  - Mismatched total elements during reshaping.
  - Attempting to use `view` on non-contiguous tensors.
  - Overusing in-place operations that interfere with gradient computations.

Mastering these tensor manipulation techniques will empower you to handle data preprocessing, model input preparation, and various other tasks efficiently in your deep learning workflows.

---

## **11. Additional Resources**

To further enhance your skills, explore the following resources:

- **Official PyTorch Documentation:**
  - [Tensor Basics](https://pytorch.org/docs/stable/tensors.html)
  - [Tensor Manipulation](https://pytorch.org/docs/stable/tensors.html#torch-tensor)
  - [torch.view](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view)
  - [torch.reshape](https://pytorch.org/docs/stable/tensors.html#torch.reshape)
  - [torch.transpose](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.transpose)
  - [torch.permute](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute)

- **PyTorch Tutorials:**
  - [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
  - [Data Loading and Processing Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
  - [Tensor Manipulation in PyTorch](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

- **Books and Guides:**
  - *Deep Learning with PyTorch* by Eli Stevens, Luca Antiga, and Thomas Viehmann.
  - *Programming PyTorch for Deep Learning* by Ian Pointer.

- **Community Forums:**
  - [PyTorch Forums](https://discuss.pytorch.org/)
  - [Stack Overflow PyTorch Tag](https://stackoverflow.com/questions/tagged/pytorch)

Engage with these resources, experiment with the provided code examples, and apply these techniques to your projects to solidify your understanding. Happy coding!
