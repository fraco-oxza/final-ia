# Wine Quality Classification: Neural Network Training Report

## Abstract

This report presents a neural network implementation from scratch using NumPy for classifying wine quality based on physicochemical properties. The model achieves approximately **59% test accuracy** on a challenging 7-class classification problem with highly imbalanced classes. We detail the architecture choices, training methodology, regularization techniques, and provide a comprehensive analysis of the results along with justifications for each design decision.

## 1. Introduction

### 1.1 Problem Statement

The goal is to predict wine quality (scores 3-9) based on 11 physicochemical features from the UCI Wine Quality dataset. This is a multi-class classification problem with:

- **6,497 samples** (combined red and white wines)
- **11 input features** (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol)
- **7 output classes** (quality scores 3, 4, 5, 6, 7, 8, 9)

### 1.2 Dataset Characteristics

The dataset presents significant challenges:

| Quality | Percentage |
|---------|------------|
| 3       | 0.46%      |
| 4       | 3.32%      |
| 5       | 32.91%     |
| 6       | 43.65%     |
| 7       | 16.61%     |
| 8       | 2.97%      |
| 9       | 0.08%      |

The severe class imbalance (classes 5 and 6 dominate with ~77% of samples) makes this a particularly difficult classification task. This imbalance is inherent to the problem domain: most wines are of average quality, while truly exceptional or poor wines are rare.

### 1.3 Why This Problem is Challenging

Wine quality prediction is inherently difficult for several reasons:

1. **Subjective Labels**: Quality scores come from human wine tasters whose opinions may vary by 1-2 points
2. **Limited Features**: The 11 physicochemical measurements cannot capture all aspects that influence perceived quality (grape variety, terroir, aging process, etc.)
3. **Ordinal Nature**: Quality is ordinal, not categorical - a score of 5 is "closer" to 6 than to 3
4. **Class Overlap**: Wines of adjacent quality levels may have very similar chemical compositions

## 2. Network Architecture

### 2.1 Layer Structure

We implemented a fully-connected feedforward neural network with the following architecture:

![Network Architecture](runs/2025-12-05_10-23-51/plot_architecture.png)

```
Input Layer:    11 neurons (one per feature)
Hidden Layer 1: 128 neurons + Leaky ReLU + Dropout
Hidden Layer 2: 64 neurons + Leaky ReLU + Dropout  
Hidden Layer 3: 32 neurons + Leaky ReLU + Dropout
Output Layer:   7 neurons + Softmax
```

**Total Architecture:** [11, 128, 64, 32, 7]

#### Why This Architecture?

We chose a **progressively narrowing architecture** (128 -> 64 -> 32) for the following reasons:

1. **Feature Abstraction**: The first layer (128 neurons) has high capacity to learn diverse low-level feature combinations from the 11 inputs. Subsequent layers progressively compress these into more abstract representations.

2. **Computational Efficiency**: A funnel shape reduces parameters compared to constant-width layers while maintaining representational power.

3. **Three Hidden Layers**: We found empirically that fewer layers (1-2) underfitted the data, while more layers (4+) provided no accuracy improvement and increased training time. Three layers provide sufficient depth to learn hierarchical features without excessive complexity.

4. **Layer Sizes**: Starting with 128 neurons gives approximately 10x expansion from the input dimension, allowing the network to learn rich combinations. The 7-neuron output matches our 7 quality classes.

### 2.2 Activation Functions

#### Leaky ReLU (Hidden Layers)

$$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

Where $\alpha = 0.01$.

**Why Leaky ReLU instead of standard ReLU?**

Standard ReLU ($f(x) = max(0, x)$) can cause "dying neurons" - when a neuron's weights are updated such that its output is always negative, the gradient becomes permanently zero, and the neuron stops learning. Leaky ReLU prevents this by allowing a small gradient ($\alpha = 0.01$) for negative inputs, ensuring all neurons remain trainable throughout the optimization process.

We chose $\alpha = 0.01$ as it's a well-established default that provides enough gradient flow without significantly altering the behavior for positive inputs.

#### Softmax (Output Layer)

$$\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Why Softmax?**

Softmax is the natural choice for multi-class classification because:

1. It converts raw logits into a valid probability distribution (outputs sum to 1)
2. It pairs mathematically with cross-entropy loss, resulting in a clean gradient: $\frac{\partial L}{\partial z} = \hat{y} - y$
3. The output probabilities are interpretable as class confidences

### 2.3 Weight Initialization

We use **He initialization**:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

**Why He Initialization?**

Proper weight initialization is critical for training deep networks. He initialization was specifically designed for ReLU-family activations. The $\sqrt{2/n_{in}}$ scaling factor accounts for the fact that ReLU zeros out roughly half the neurons, so we need larger initial weights to maintain signal variance through the network.

Using naive initialization (e.g., small random values) with deep networks leads to:

- **Vanishing activations**: Signal diminishes exponentially through layers
- **Exploding gradients**: Gradients grow uncontrollably during backpropagation

He initialization prevents both issues, enabling stable training from the start.

## 3. Training Methodology

### 3.1 Loss Function

**Cross-Entropy Loss** for multi-class classification:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

**Why Cross-Entropy?**

Cross-entropy is the standard loss for classification because:

1. **Information-theoretic foundation**: It measures the "surprise" of predictions relative to true labels
2. **Gradient properties**: Combined with softmax, it produces well-behaved gradients that are large when predictions are wrong and small when correct
3. **Probabilistic interpretation**: Minimizing cross-entropy is equivalent to maximum likelihood estimation

Alternative losses like MSE perform poorly for classification because they don't properly penalize confident wrong predictions.

### 3.2 Optimizer: AdamW

We implemented **AdamW** (Adam with decoupled weight decay), chosen over simpler optimizers for its superior convergence properties.

**Update equations:**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

**Why AdamW over SGD or standard Adam?**

1. **vs. SGD**: Adam adapts the learning rate per-parameter based on historical gradients. This is crucial for our problem where different features (and their corresponding weights) may require different learning rates. SGD with a single global learning rate often requires extensive hyperparameter tuning.

2. **vs. Adam**: Standard Adam applies weight decay to the gradient before the adaptive scaling, which inadvertently couples regularization strength with the learning rate. AdamW decouples these, applying weight decay directly to weights. This results in more consistent regularization across parameters.

**Hyperparameter Choices:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning rate | 0.002 | Slightly higher than default (0.001) since dropout slows convergence |
| Beta 1 | 0.9 | Standard value, provides momentum over ~10 iterations |
| Beta 2 | 0.999 | Standard value, smooths learning rate adaptation |
| Epsilon | 1e-8 | Prevents division by zero |
| Weight decay | 0.001 | Mild regularization to complement dropout |

### 3.3 Regularization Techniques

#### Dropout (30%)

During training, we randomly zero out 30% of neurons in each hidden layer.

**Why Dropout?**

Our initial experiments without dropout showed severe overfitting: ~77% training accuracy but only ~56% test accuracy (21% gap). Dropout addresses this by:

1. **Preventing co-adaptation**: Neurons cannot rely on specific other neurons, forcing them to learn more robust features
2. **Implicit ensemble**: Dropout approximately trains an ensemble of $2^n$ different networks (where n = number of neurons), with the final model averaging their predictions
3. **Noise injection**: Acts as a form of data augmentation at the feature level

**Why 30%?**

We tested dropout rates of 0%, 20%, 30%, 40%, and 50%. 30% achieved the best balance:
- 0-20%: Still significant overfitting
- 30%: Train-test gap reduced to ~4%
- 40-50%: Underfitting, lower overall accuracy

**Inverted Dropout Implementation**:

We scale activations by $\frac{1}{1-p}$ during training rather than scaling at test time. This keeps inference efficient and matches modern framework implementations.

#### Early Stopping

Training stops when validation loss doesn't improve for 300 consecutive epochs, restoring weights from the best epoch.

**Why Early Stopping?**

Neural networks will eventually memorize training data if trained long enough. Early stopping provides:

1. **Automatic regularization**: Limits model complexity by stopping before overfitting
2. **Computational efficiency**: Avoids wasting compute on unproductive epochs
3. **No hyperparameter tuning**: Unlike explicit regularization (L2, dropout rate), early stopping adapts automatically

**Why 300 epochs patience?**

Our loss curves showed that validation loss can stagnate for 100-200 epochs before improving. A patience of 300 ensures we don't stop prematurely while still catching true plateaus.

#### Learning Rate Decay

$$\eta_{new} = 0.95 \times \eta_{old} \text{ every 500 epochs}$$

**Why Learning Rate Decay?**

Large learning rates help escape local minima early in training but cause oscillation near optima. Gradually reducing the learning rate allows:

1. **Initial exploration**: High LR explores the loss landscape broadly
2. **Fine-tuning**: Lower LR allows precise convergence to minima

The decay factor (0.95) and schedule (every 500 epochs) were chosen to reduce LR by roughly 50% over the typical training duration (~2000-3000 epochs before early stopping).

### 3.4 Data Preprocessing

#### Z-score Normalization

Each feature is standardized: $x' = \frac{x - \mu}{\sigma}$

**Why Normalize?**

Our features have vastly different scales:
- Density: ~0.99-1.04
- Total sulfur dioxide: ~6-440
- Alcohol: ~8-15

Without normalization:
1. Large-scale features dominate gradient updates
2. The loss landscape becomes elongated, slowing convergence
3. Learning rate must be set for the largest-scale feature, causing slow learning for others

Z-score normalization puts all features on comparable scales (mean=0, std=1), enabling faster and more stable training.

#### Stratified Train/Validation/Test Split

Data is split 70%/15%/15% while **maintaining class proportions** in each split.

**Why Stratified Splitting?**

With severe class imbalance, random splitting could put all samples of class 9 (only 5 samples total) into the training set, leaving none for validation/test. Stratified splitting ensures:

1. Each split has the same class distribution as the full dataset
2. Validation and test metrics are representative of true performance
3. Rare classes are fairly evaluated

## 4. Training Results

### 4.1 Training History

![Training History](runs/2025-12-05_10-23-51/plot_training_history.png)

**Observations:**

1. **Convergence**: Training loss decreases steadily, indicating the optimization is working correctly
2. **Generalization Gap**: The gap between training and validation curves shows controlled overfitting (~6-7% accuracy gap)
3. **Early Stopping**: Training was terminated at epoch ~1282 when validation loss stopped improving, with best weights restored from epoch 982
4. **Learning Rate Effect**: Small jumps in the curves every 500 epochs correspond to learning rate decay

### 4.2 Final Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 63.03% |
| Validation Accuracy | 56.95% |
| Test Accuracy | **58.91%** |

The test accuracy (58.91%) being between training and validation suggests:
- The model generalizes reasonably well
- Early stopping successfully prevented severe overfitting
- The validation set was representative of test performance

### 4.3 Class Distribution Analysis

![Class Distribution](runs/2025-12-05_10-23-51/plot_class_distribution.png)

**Observations:**

The model's prediction distribution differs from the true distribution:
- **Over-predicts classes 5 and 6**: The model is biased toward majority classes
- **Under-predicts extreme classes (3, 4, 8, 9)**: Rare classes receive almost no predictions

This behavior is expected without explicit class balancing. The model learns that predicting "6" is usually correct (44% of the time), so it favors this safe prediction.

## 5. Model Evaluation

### 5.1 Confusion Matrices

We present three views of the confusion matrix to understand different aspects of model performance.

#### Raw Counts
![Confusion Matrix Raw](runs/2025-12-05_10-23-51/plot_confusion_matrix_raw.png)

The raw confusion matrix shows actual prediction counts. Key observations:
- The diagonal (correct predictions) is most populated for classes 5, 6, 7
- Classes 3 and 9 have almost no predictions at all
- Most errors are off-diagonal by only 1 class (e.g., true=5, predicted=6)

#### Normalized by True Label
![Confusion Matrix Recall](runs/2025-12-05_10-23-51/plot_confusion_matrix_recall.png)

This view shows, for each true class, what fraction was predicted as each class. It reveals:
- Class 6 has the highest recall (~65%): the model correctly identifies most quality-6 wines
- Classes 3 and 9 have near-zero recall: the model essentially never predicts these
- Adjacent class confusion is evident (5-6, 6-7 are often confused)

#### Normalized by Prediction
![Confusion Matrix Precision](runs/2025-12-05_10-23-51/plot_confusion_matrix_precision.png)

This view shows, for each predicted class, what fraction was correct. Columns for classes 3 and 9 are empty because the model never predicts them.

### 5.2 Per-Class Metrics

![Per-Class Metrics](runs/2025-12-05_10-23-51/plot_per_class_metrics.png)

The per-class breakdown confirms:
- **Best performance**: Classes 5, 6, 7 (sufficient training data)
- **Poor performance**: Classes 3, 4, 8, 9 (insufficient data and class imbalance)

### 5.3 Prediction Confidence

![Prediction Confidence](runs/2025-12-05_10-23-51/plot_confidence.png)

**Analysis:**

- Correct predictions tend to have higher confidence (softmax probability)
- However, many incorrect predictions also show high confidence
- This indicates the model is sometimes confidently wrong, likely on ambiguous samples near class boundaries

## 6. Discussion

### 6.1 Why ~59% Accuracy is Reasonable

Our 59% test accuracy may seem low, but context is essential:

1. **Baseline Comparison**: Random guessing achieves 14.3% accuracy (1/7 classes). Our model achieves **4.1x** this baseline.

2. **Majority Class Baseline**: Always predicting class 6 achieves 43.65% accuracy. Our model improves by **15 percentage points** over this naive strategy.

3. **Literature Comparison**: Published results on this dataset using sophisticated methods (SVM, Random Forest, XGBoost) typically achieve 55-65% accuracy. Our from-scratch neural network is competitive.

4. **Problem Difficulty**: Adjacent wine quality levels (e.g., 5 vs 6) may be indistinguishable even to human experts. The chemical features simply may not contain enough information to reliably separate all classes.

### 6.2 Techniques That Helped

| Technique | Impact | Evidence |
|-----------|--------|----------|
| AdamW optimizer | Faster convergence | Converged in ~1000 epochs vs ~3000+ with SGD |
| Dropout (30%) | Reduced overfitting | Train-test gap: 22% -> 4% |
| Early Stopping | Optimal weights | Restored from epoch 982, avoiding later overfitting |
| Stratified Split | Fair evaluation | All classes represented in test set |
| He Initialization | Stable training | No gradient issues from epoch 1 |
| Learning Rate Decay | Fine-tuned convergence | Smoother loss curves in later epochs |

### 6.3 Limitations

1. **Class Imbalance**: The model effectively ignores classes 3 and 9 due to their extreme rarity. This is a fundamental limitation without explicit handling (class weights, oversampling).

2. **Feature Limitations**: Only 11 physicochemical features are available. Wine quality depends on many factors not captured (grape variety, vintage, winemaking process, taster preferences).

3. **Ordinal vs Categorical**: We treat wine quality as 7 independent classes, but in reality, quality is ordinal. A prediction of 6 for a true label of 5 is more reasonable than predicting 3, but our loss treats both errors equally.

### 6.4 Potential Improvements

1. **Class Weighting**: Assign higher loss penalties to minority classes, forcing the model to learn them despite limited examples.

2. **Oversampling (SMOTE)**: Generate synthetic samples for rare classes to balance the training distribution.

3. **Ordinal Regression**: Use a loss function that penalizes predictions based on ordinal distance (e.g., predicting 3 for true 9 should cost more than predicting 8).

4. **Feature Engineering**: Create polynomial features, interaction terms, or domain-specific ratios that may be more predictive.

5. **Ensemble Methods**: Train multiple models with different initializations or architectures and combine predictions.

## 7. Conclusion

We successfully implemented a neural network from scratch using only NumPy that achieves **~59% test accuracy** on the challenging Wine Quality classification task. The implementation demonstrates understanding of:

- Forward propagation with Leaky ReLU activations and Softmax output
- Backpropagation with manually derived gradients
- AdamW optimization with momentum, adaptive learning rates, and decoupled weight decay
- Regularization through Dropout, Early Stopping, and Learning Rate Decay
- Data handling with stratified splitting and Z-score normalization

The results are reasonable given the dataset challenges. The model reliably classifies wines of average quality (scores 5-7) but struggles with rare extreme scores, a limitation inherent to the class-imbalanced nature of the problem.

The ~59% accuracy significantly outperforms random chance (14%) and the majority-class baseline (44%), demonstrating that the neural network successfully learned meaningful patterns from the physicochemical features.

## References

1. Cortez, P., et al. (2009). "Modeling wine preferences by data mining from physicochemical properties." Decision Support Systems.
2. UCI Machine Learning Repository: Wine Quality Dataset (ID: 186)
3. Kingma, D.P. & Ba, J. (2014). "Adam: A Method for Stochastic Optimization."
4. Loshchilov, I. & Hutter, F. (2017). "Decoupled Weight Decay Regularization."
5. He, K., et al. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification."
6. Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting."
