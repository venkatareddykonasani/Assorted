### The Data

| Input \(x_1\) | Input \(x_2\) | Output \(y\) |
|---------------|---------------|--------------|
| 0             | 0             | 0            |
| 0             | 1             | 1            |
| 1             | 0             | 1            |
| 1             | 1             | 0            |

### Neural Network Architecture

- **Input Layer**: Two inputs (\(x_1\), \(x_2\))
- **Hidden Layer**: Two neurons
- **Output Layer**: One neuron

### Step-by-Step Walkthrough

1. **Initialization**
   - Randomly initialize weights for:
     - Input to Hidden Layer: \(w_1, w_2, w_3, w_4\)
     - Hidden to Output Layer: \(w_5, w_6\)
   - Assume biases for hidden and output layers are also initialized as \(b_1, b_2\) for hidden neurons, and \(b_3\) for the output neuron.
   - Set learning rate (\(\alpha\)) to a small value, e.g., 0.1.

   Let’s initialize:
   - \(w_1 = 0.1\), \(w_2 = 0.2\), \(w_3 = 0.3\), \(w_4 = 0.4\)
   - \(w_5 = 0.5\), \(w_6 = 0.6\)
   - \(b_1 = 0.1\), \(b_2 = 0.1\), \(b_3 = 0.1\)

2. **Forward Propagation**

   We will run forward propagation for a single data point, \((x_1, x_2) = (0, 1)\), and target output \(y = 1\):

   - **Hidden Layer Calculations**:

     \[
     z_1 = w_1 x_1 + w_2 x_2 + b_1 = (0.1 \times 0) + (0.2 \times 1) + 0.1 = 0.3
     \]

     \[
     h_1 = \text{sigmoid}(z_1) = \frac{1}{1 + e^{-0.3}} \approx 0.5744
     \]

     \[
     z_2 = w_3 x_1 + w_4 x_2 + b_2 = (0.3 \times 0) + (0.4 \times 1) + 0.1 = 0.5
     \]

     \[
     h_2 = \text{sigmoid}(z_2) = \frac{1}{1 + e^{-0.5}} \approx 0.6225
     \]

   - **Output Layer Calculations**:

     \[
     z_3 = w_5 h_1 + w_6 h_2 + b_3 = (0.5 \times 0.5744) + (0.6 \times 0.6225) + 0.1
     \]

     \[
     z_3 \approx 0.2872 + 0.3735 + 0.1 = 0.7607
     \]

     \[
     \hat{y} = \text{sigmoid}(z_3) = \frac{1}{1 + e^{-0.7607}} \approx 0.6815
     \]

3. **Calculate Loss**
   
   The **loss** is calculated using the **Mean Squared Error (MSE)**:

   \[
   L = \frac{1}{2} (y - \hat{y})^2 = \frac{1}{2} (1 - 0.6815)^2 \approx 0.0507
   \]

4. **Backpropagation**

   We need to calculate the gradient of the loss function with respect to each weight using the **chain rule**.

   - **Output Layer Gradients**:

     For weight \(w_5\):

     \[
     \frac{\partial L}{\partial w_5} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} \cdot \frac{\partial z_3}{\partial w_5}
     \]

     - \(\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 0.6815 - 1 = -0.3185\)
     - \(\frac{\partial \hat{y}}{\partial z_3} = \hat{y} (1 - \hat{y}) = 0.6815 \times (1 - 0.6815) \approx 0.2170\)
     - \(\frac{\partial z_3}{\partial w_5} = h_1 = 0.5744\)

     Thus:

     \[
     \frac{\partial L}{\partial w_5} = (-0.3185) \times 0.2170 \times 0.5744 \approx -0.0397
     \]

     Similarly, calculate the gradients for \(w_6\) and the bias \(b_3\).

   - **Hidden Layer Gradients**:

     For weight \(w_1\):

     \[
     \frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} \cdot \frac{\partial z_3}{\partial h_1} \cdot \frac{\partial h_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}
     \]

     - \(\frac{\partial L}{\partial \hat{y}} = -0.3185\)
     - \(\frac{\partial \hat{y}}{\partial z_3} = 0.2170\)
     - \(\frac{\partial z_3}{\partial h_1} = w_5 = 0.5\)
     - \(\frac{\partial h_1}{\partial z_1} = h_1 (1 - h_1) = 0.5744 \times (1 - 0.5744) \approx 0.2445\)
     - \(\frac{\partial z_1}{\partial w_1} = x_1 = 0\)

     Since \(x_1 = 0\), \(\frac{\partial L}{\partial w_1} = 0\), which means no update for \(w_1\) in this iteration for this specific data point.

5. **Weight Update**

   Using **gradient descent**, update the weights:

   \[
   w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial L}{\partial w}
   \]

   For \(w_5\):

   \[
   w_5 = w_5 - 0.1 \times (-0.0397) = 0.5 + 0.00397 = 0.50397
   \]

   Similarly, update \(w_6, w_1, w_2, w_3, w_4\), and biases \(b_1, b_2, b_3\).

6. **Repeat for All Data Points**
   - Repeat the above steps for each training example in the XOR truth table.
   - Perform multiple **epochs** (iterations over the entire dataset) until the loss is minimized and the network learns the correct XOR outputs.

### Summary
- **Forward Propagation**: Calculate the predicted output using the current weights and activation functions.
- **Calculate Loss**: Measure the difference between the predicted and actual values.
- **Backward Propagation (Using Chain Rule)**: Calculate gradients of the loss function with respect to each weight by propagating errors backward through the network using the chain rule.
- **Gradient Descent**: Update the weights to minimize the loss function.

By repeating this process, the ANN learns the correct mapping for the XOR function, effectively finding the best weights that produce the desired output for each combination of inputs.
