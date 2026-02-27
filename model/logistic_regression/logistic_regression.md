# Logistic Regression — Model Documentation

## Overview
This implementation trains a logistic regression model from scratch to classify whether a student passes an exam based on hours studied. Unlike linear regression (which predicts continuous values), logistic regression predicts probabilities and classifies them into binary outcomes (pass/fail).

- **Model equation:**
  $$f(x) = \sigma(wx + b)$$
  where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.
- **Output:** Probability between 0 and 1 (interpreted as the chance of passing).
- **Classification rule:** Probability $\geq 0.5$ → class 1 (pass), else class 0 (fail).

## Key Components

### 1. Sigmoid Function
Squashes any real number into the (0, 1) range, giving a probabilistic interpretation to the model's output.

### 2. Cost Function (Log Loss / Binary Cross-Entropy)
Measures how wrong the model is:
$$J(w, b) = -\frac{1}{m} \sum [y \log(p) + (1-y) \log(1-p)]$$
where $p$ is the predicted probability. This cost penalizes confident wrong predictions heavily, ensuring the model learns meaningful boundaries.

### 3. Gradient Computation
Calculates the slope of the cost function with respect to $w$ and $b$:
- $\frac{\partial J}{\partial w} = \frac{1}{m} \sum (p - y)x$
- $\frac{\partial J}{\partial b} = \frac{1}{m} \sum (p - y)$

These gradients guide how to update $w$ and $b$ to reduce the cost.

### 4. Gradient Descent
Iteratively updates $w$ and $b$ using the gradients to minimize the cost function:
- $w := w - \alpha \frac{\partial J}{\partial w}$
- $b := b - \alpha \frac{\partial J}{\partial b}$

where $\alpha$ is the learning rate.

### 5. Prediction and Classification
- **predict:** Computes probabilities for new data using the trained $w$ and $b$.
- **classify:** Converts probabilities to hard class labels (0 or 1) using a threshold (default 0.5).

### 6. Accuracy
Measures the fraction of correct predictions on test data.

## Training and Evaluation Flow
1. **Initialize** $w$ and $b$ to zero.
2. **Train** using gradient descent for a set number of iterations, updating $w$ and $b$ each time.
3. **Monitor** cost history to ensure the model is learning (cost should decrease).
4. **Test** on unseen data:
   - Predict probabilities and classify.
   - Compute accuracy.
5. **Visualize:**
   - Plot the sigmoid decision curve and training/test data.
   - Plot the cost history (learning curve).

## Example Output
- Trained parameters: $w$, $b$
- Test accuracy: e.g., 100%
- Cost at start/end: Shows learning progress
- Plots: Sigmoid curve with data points, cost history

## Notes
- All math (sigmoid, log-loss, gradients) is implemented from scratch.
- The model is robust to edge cases (e.g., avoids log(0) errors).
- Visualization is for interpretability; it does not affect learning.

---

**See `logistic_regression.py` for full code and inline explanations.**
