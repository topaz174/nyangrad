# Softmax Regression Cheat Sheet

## 1. Fundamentals

### Logits and Softmax Function
For a multi-class classification problem with $k$ classes, the **logits** $z \in \mathbb{R}^k$ are the raw scores predicted for each class.
The **Softmax function** converts logits into probabilities:
$$z_i = \mathbb{P}(y=i | x) = \frac{\exp(h_i)}{\sum_{j=1}^k \exp(h_j)}$$
where $h_i$ is the logit for class $i$.

### Softmax Loss (Cross-Entropy Loss)
For a single sample with logits $z$ and true label $y \in \{1, \dots, k\}$:
$$\ell_{\text{softmax}}(z, y) = \log \sum_{j=1}^k \exp z_j - z_y$$

## 2. Linear Softmax Regression

### Hypothesis
The hypothesis function maps $n$-dimensional inputs to $k$-dimensional logits via a linear transformation:
$$h(x) = \Theta^T x$$
where:
- $x \in \mathbb{R}^n$ is the input vector.
- $\Theta \in \mathbb{R}^{n \times k}$ is the parameter matrix.

### Optimization Problem
Minimize the average softmax loss over $m$ samples:
$$\min_{\Theta} \frac{1}{m} \sum_{i=1}^m \ell_{\text{softmax}}(\Theta^T x^{(i)}, y^{(i)})$$

## 3. Gradients and Updates

### Single Sample Gradient
$$\nabla_{\Theta} \ell_{\text{softmax}}(\Theta^T x, y) = x (z - e_y)^T$$
where:
- $z = \text{normalize}(\exp(\Theta^T x))$ (softmax probabilities).
- $e_y$ is the one-hot vector for label $y$.

### Batch Gradient (Matrix Form)
Given a batch $X \in \mathbb{R}^{m \times n}$ and labels $y \in \{1, \dots, k\}^m$:
$$\nabla_{\Theta} \ell_{\text{softmax}}(X \Theta, y) = \frac{1}{m} X^T (Z - I_y)$$
where:
- $Z = \text{normalize}(\exp(X \Theta))$ (row-wise softmax).
- $I_y \in \mathbb{R}^{m \times k}$ is the one-hot encoding matrix of labels $y$.

### SGD Update Rule
$$\Theta := \Theta - \alpha \cdot \nabla_{\Theta} \mathcal{L}$$
where $\alpha$ is the learning rate.

## 4. Matrix Dimensions Reference
| Variable | Notation | Dimensions | Description |
| :--- | :--- | :--- | :--- |
| **Input** | $X$ | $m \times n$ | $m$ samples, $n$ features |
| **Weights** | $\Theta$ | $n \times k$ | $n$ features, $k$ classes |
| **Logits** | $H = X\Theta$ | $m \times k$ | Batch of raw scores |
| **Probs** | $Z = \text{softmax}(H)$ | $m \times k$ | Row-wise probabilities |
| **Labels** | $y$ | $m \times 1$ | True class indices |
| **One-hot** | $I_y$ | $m \times k$ | One-hot binary matrix |
