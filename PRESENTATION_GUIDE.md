# Presentation Guide: Wine Quality Neural Network

## Suggested Structure: 10-12 slides, ~15 minutes

---

## Slide 1: Title

**On the slide:**
- Wine Quality Classification with Neural Networks
- Implementation from scratch with NumPy
- Your name, date, course

**What to say:**
> "Today I'm going to present a project where I implemented a neural network completely from scratch, using only NumPy, to classify wine quality based on their chemical properties."

---

## Slide 2: The Problem

**On the slide:**
- Dataset: UCI Wine Quality (6,497 samples)
- 11 chemical features → 7 quality classes (3-9)
- Class distribution table showing the imbalance

| Quality | Percentage |
|---------|------------|
| 3       | 0.46%      |
| 4       | 3.32%      |
| 5       | 32.91%     |
| 6       | 43.65%     |
| 7       | 16.61%     |
| 8       | 2.97%      |
| 9       | 0.08%      |

**What to say:**
> "The dataset has almost 6,500 wines with 11 chemical measurements like acidity, residual sugar, pH, alcohol, etc. The goal is to predict quality from 3 to 9."

> "But there's an important problem: the classes are very imbalanced. 77% of wines are quality 5 or 6. Extreme qualities like 3 or 9 are extremely rare - there are only 5 wines of quality 9 in the entire dataset."

> "This makes the problem very difficult because the model tends to ignore rare classes."

---

## Slide 3: Why is it difficult?

**On the slide:**
- Extreme class imbalance
- Subjective labels (human tasters)
- Adjacent classes chemically very similar
- Limited features (don't capture grape variety, aging, etc.)

**What to say:**
> "Besides the imbalance, there are other challenges. The scores come from human tasters who may differ by 1 or 2 points. A quality 5 wine and a quality 6 wine can have almost identical chemical compositions."

> "And we only have 11 chemical variables - we don't know the grape variety, the winemaking process, or anything sensory. So there's a limit to how well we can predict."

---

## Slide 4: Network Architecture

**On the slide:**
- Image: `plot_architecture.png`
- Structure: [11, 128, 64, 32, 7]
- Activations: Leaky ReLU + Softmax

**What to say:**
> "I designed a network with 3 hidden layers that gradually narrows: 128, 64, 32 neurons. This allows the first layer to learn many feature combinations, and the following layers compress them into more abstract representations."

> "I used Leaky ReLU instead of regular ReLU because ReLU can cause 'dead neurons' - if a neuron always outputs negative, its gradient is zero and it stops learning. Leaky ReLU prevents this by allowing a small gradient for negative values."

> "The output layer uses Softmax to convert logits into probabilities over the 7 classes."

---

## Slide 5: Design Decisions - Initialization and Loss

**On the slide:**
- He Initialization: $W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$
- Cross-Entropy Loss: $\mathcal{L} = -\sum y \log(\hat{y})$

**What to say:**
> "For the weights I used He initialization, designed specifically for ReLU-type activations. If you initialize with very small values, the signal vanishes layer by layer. If they're too large, gradients explode. He initialization keeps the variance stable."

> "As loss function I used cross-entropy, which is standard for classification. The advantage is that when you combine it with softmax, the gradient becomes very clean: simply prediction minus true label."

---

## Slide 6: Optimizer - AdamW

**On the slide:**
- AdamW = Adam + Decoupled Weight Decay
- Momentum + Adaptive learning rates
- Hyperparameters: lr=0.002, β1=0.9, β2=0.999, weight_decay=0.001

**What to say:**
> "Instead of simple SGD, I implemented AdamW. Adam combines momentum - which accelerates convergence in consistent directions - with adaptive learning rates per parameter."

> "The 'W' stands for decoupled weight decay. In regular Adam, weight decay is applied to the gradient before adaptive scaling, which mixes regularization with learning rate. AdamW separates them, applying decay directly to the weights."

> "This allowed me to converge in ~1000 epochs versus ~3000+ that I needed with basic SGD."

---

## Slide 7: Regularization

**On the slide:**
- **Dropout 30%**: Prevents co-adaptation
- **Early Stopping**: Patience of 300 epochs
- **Learning Rate Decay**: 0.95x every 500 epochs

**What to say:**
> "Overfitting was a serious problem. Without regularization, I had 77% training accuracy but only 56% test - a 21 point gap."

> "Dropout helped a lot: during training, I randomly turn off 30% of neurons. This forces the network not to rely on specific neurons and learn more robust features. I tested various values and 30% gave the best balance."

> "Early stopping halts training when validation loss stops improving for 300 epochs, and restores the best weights. This prevents the model from memorizing the training set."

> "I also gradually reduce the learning rate for fine-tuning at the end of training."

---

## Slide 8: Preprocessing

**On the slide:**
- Z-score normalization: $x' = (x - \mu) / \sigma$
- Stratified split: 70% train, 15% val, 15% test
- Maintains class distribution in each split

**What to say:**
> "The features have very different scales - density goes from 0.99 to 1.04, but sulfur dioxide goes from 6 to 440. Without normalizing, larger-scale features dominate training."

> "To split the data I used stratified split, which maintains the proportion of each class in train, validation and test. This is critical with such imbalanced classes - without this, I could end up with no class 9 examples in the test set."

---

## Slide 9: Results - Training Curves

**On the slide:**
- Image: `plot_training_history.png`
- Early stopping at epoch ~1282, restored epoch 982

**What to say:**
> "Here you can see the training curves. Training loss decreases consistently, while validation loss stabilizes around epoch 1000."

> "Early stopping detected there was no improvement and stopped at epoch 1282, restoring the weights from epoch 982 where the best validation loss was."

> "The gap between training and validation curves shows controlled overfitting - thanks to dropout and early stopping, it's only ~6% instead of 21%."

---

## Slide 10: Results - Final Metrics

**On the slide:**
| Metric | Value |
|--------|-------|
| Training Accuracy | 63.03% |
| Validation Accuracy | 56.95% |
| Test Accuracy | **58.91%** |

Comparison with baselines:
- Random: 14.3%
- Always predict class 6: 43.65%
- **Our model: 58.91%**

**What to say:**
> "The model achieved almost 59% test accuracy. It may seem low, but we need to put it in context."

> "Random guessing would give 14% - we have 4 times that. Always predicting the majority class would give 44% - we beat that by 15 points."

> "In the literature, sophisticated methods like Random Forest or XGBoost on this dataset achieve between 55-65%. Our neural network from scratch is competitive."

---

## Slide 11: Error Analysis

**On the slide:**
- Image: `plot_confusion_matrix_recall.png` (normalized by row)
- Model predicts classes 5, 6, 7 well
- Almost never predicts classes 3 and 9

**What to say:**
> "The confusion matrix reveals the model's behavior. The diagonal shows correct predictions."

> "The model works well for qualities 5, 6 and 7 where there's enough data. But it almost never predicts extreme classes 3 and 9."

> "This is expected without class balancing - the model learns that predicting 6 is almost always a safe bet."

> "Also notice that most errors are between adjacent classes - it confuses 5 with 6, or 6 with 7. This makes sense because wines of close qualities are chemically similar."

---

## Slide 12: Limitations and Future Improvements

**On the slide:**
**Limitations:**
- Doesn't handle rare classes well (3, 9)
- Chemical features don't capture everything
- Treats quality as categorical, not ordinal

**Possible improvements:**
- Class weighting
- Oversampling (SMOTE)
- Ordinal regression
- More features

**What to say:**
> "The model has clear limitations. The main one is that it ignores very rare classes because it doesn't have enough examples to learn them."

> "We also treat quality as 7 independent classes, but in reality it's ordinal - being wrong by 1 point should be penalized less than being wrong by 5."

> "To improve, we could use class weighting to penalize errors on minority classes more, or generate synthetic data with SMOTE. We could also use ordinal regression that respects the natural order of classes."

---

## Slide 13: Conclusion

**On the slide:**
- Neural network implemented 100% from scratch with NumPy
- ~59% accuracy (4x better than random, 15pp better than baseline)
- Key techniques: AdamW, Dropout, Early Stopping, Stratified Split
- Code available at: [repo link]

**What to say:**
> "In summary, I implemented a complete neural network from scratch - forward propagation, backpropagation, AdamW, dropout, everything manually with NumPy."

> "The model achieves almost 59% accuracy on a difficult problem with 7 imbalanced classes, significantly outperforming simple baselines."

> "Regularization techniques were key to controlling overfitting, and stratified split ensured fair evaluation."

> "Any questions?"

---

## Presentation Tips

1. **Practice transitions** between slides - each one should flow naturally to the next.

2. **Prepare for common questions:**
   - "Why didn't you use PyTorch/TensorFlow?" → The goal was to understand the fundamentals by implementing from scratch.
   - "How would you improve accuracy?" → Class weighting, SMOTE, ordinal regression, ensemble methods.
   - "Why 3 layers and not more?" → I tested 1-5 layers, 3 gave the best balance between capacity and generalization.
   - "Why AdamW and not SGD?" → AdamW converges faster and has better regularization.

3. **If you have time**, you can do a live demo running the script and showing the output.

4. **Emphasize** that everything was implemented from scratch - that's what makes the project impressive.

5. **Suggested time per slide:**
   - Slides 1-3: 2-3 minutes (intro and problem)
   - Slides 4-8: 5-6 minutes (architecture and methodology)
   - Slides 9-11: 4-5 minutes (results)
   - Slides 12-13: 2-3 minutes (limitations and conclusion)
