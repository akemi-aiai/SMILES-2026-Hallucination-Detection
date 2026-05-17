# Hallucination Detection Project Submission
Iaroslava Antoshina — SMILES-2026

Contents
1. Reproducibility Instructions
2. Final Solution Description
2.1. Feature Extraction: Multi-Layer Concatenation
2.2. Classifier: Heavily Regularized Linear Probe
2.3. Cross-Validation: Stratified K-Fold
3. Experiments and Failed Attempts
3.1 Deep MLP with AdamW and Dropout
3.2 Dimensionality Reduction via PCA
3.3 Mean Pooling across all tokens
4. Conclusion


## 1. Reproducibility Instructions

The solution was developed and tested using *Google Colab (T4 GPU)*. 
Public link to submitted predictions.csv: [Google Drive link](https://drive.google.com/file/d/1kAV7cG3UV7bGILmKcHW-WxLC0U2UhewV/view?usp=sharing)


## 2. Final Solution Description

The problem that I face is to classify 896-dimensional hidden states using only 689 labeled samples (with a 483/206 class imbalance). Standard Deep Learning approaches instantly memorize such datasets.
My final approach abandons non-linear MLPs and dimensionality reduction in favor of Linear Probing combined with Multi-Layer Representation Extraction.
### 2.1. Feature Extraction: Multi-Layer Concatenation
Instead of extracting only the hidden state of the final layer `(hidden_states[-1])`, I extracted and concatenated the hidden states of the final token from three distinct depths of the transformer: Early (Layer -12), Middle (Layer -6), and Late (Layer -1). I also extracted the L2-norm for each of these layers and the overall sequence length.
I made this choice because in the field of Mechanistic Interpretability, it is well-known that transformers process information in stages.
**Early layers (-12):** retrieve basic factual knowledge from the model's weights;  **Middle layers (-6):** form conceptual representations and logical structures; **Late layers (-1):** primarily focus on syntax and vocabulary selection for the next token. By concatenating these three stages ($896 \times 3 = 2688$ features), the classifier gets a holistic "timeline" of the LLM's thought process, capturing the exact moment a hallucination (fabrication) is introduced. The L2-norm was added because a drop in activation magnitudes often correlates with model uncertainty.
### 2.2. Classifier: Heavily Regularized Linear Probe
I completely replaced the MLP neural network in probe.py with a classical LogisticRegression model from scikit-learn (wrapped in the HallucinationProbe class to maintain compatibility).
*Hyperparameters used:* solver='liblinear', class_weight='balanced', and an extremely strict L2 penalty (C=0.005).
When the feature dimension (D = 2692) is vastly larger than the number of training samples ($N \approx 440$ per fold), any hidden layer in a neural network acts as a memorization matrix. A linear model with strict L2 regularization ($C=0.005$) forces the weights of noisy, irrelevant features to zero. It searches for a single, stable hyperplane that separates truth from hallucination, ensuring high generalizability to unseen data (Test Set). The balanced class weight corrects the bias caused by having twice as many hallucinated samples as truthful ones.### 2.3. Cross-Validation: Stratified K-Fold
* What I modified: Replaced random splitting with StratifiedKFold.
* Why I made this choice: Due to the small dataset size and class imbalance, a random split could easily create a validation fold with almost no truthful samples, leading to highly volatile and unreliable AUROC metrics. Stratification ensures the exact ~70/30 ratio is preserved across all folds.
### Biggest Contributor to the Metric:
The single largest leap in the Test AUROC metric (from ~56% to 72.30%) came from discarding the deep MLP and replacing it with the Logistic Regression Linear Probe. It stopped the catastrophic overfitting (where Train AUROC was 100%) and allowed the model to generalize.


## 3. Experiments and Failed Attempts
To arrive at the final solution, several hypotheses were empirically tested and rejected:

### 3.1 Deep MLP with AdamW and Dropout
One of my ideas was to built a standard neural network with BatchNorm1d, Dropout(0.6), and use the AdamW optimizer (which includes weight decay) to prevent overfitting. But the result was catastrophic overfitting as the model achieved a Train AUROC of 100.00% within 50 epochs, but the Test AUROC hovered around 56-57%.
The reason why it is failed is that neural networks are universal approximators. Even with heavy dropout, a 128-neuron hidden layer contains enough parameters to completely memorize 440 data points in an 896-dimensional space. The network learned the noise instead of the hallucination signal.
### 3.2 Dimensionality Reduction via PCA
Then I tried to fix the overfitting of the MLP, I applied Principal Component Analysis (sklearn.decomposition.PCA) to compress the 898 features down to 64 components before feeding them to the network. But the metric degraded even further, dropping to Test AUROC ~54%.
Later I found out that PCA maximizes variance. In LLM hidden states, the largest variance is usually driven by obvious structural features (e.g., sequence length, presence of specific punctuation, or token frequency). The "hallucination signal" is a very subtle, fine-grained semantic shift. PCA treated this vital, low-variance signal as "background noise" and discarded it entirely. This proved that we needed a classifier capable of processing all high-dimensional features directly (like Logistic Regression with L2 penalty).
### 3.3 Mean Pooling across all tokens
Another my failure instead of taking the hidden state of the final <|endoftext|> token, I averaged the hidden states of all tokens in the generated response (Mean Pooling).
As the result performance was significantly worse than extracting the exact final token across multiple depths.
It happened because averaging tokens dilutes the signal. The transformer architecture relies on attention mechanisms where later tokens aggregate context from earlier ones. Therefore, the final token's hidden state already acts as a highly refined summary vector of the entire generation step.


## 4. Conclusion

This project highlights a crucial lesson in Mechanistic Interpretability and applied Machine Learning: more complex models are not always better. When dealing with high-dimensional feature spaces ($D = 2692$) and extremely limited labeled data ($N = 689$), standard deep neural networks easily fall victim to the curse of dimensionality and memorize noise.

By pivoting to a mathematically robust, heavily regularized Linear Probe (Logistic Regression) and enriching the feature space with multi-layer representations (Layers -12, -6, and -1), the solution successfully bridged the generalization gap. The final model effectively captures the subtle semantic shifts that indicate a hallucination, achieving a highly competitive and stable **Test AUROC of 72.30%**. 

Ultimately, this approach proves that combining domain knowledge of transformer architectures (extracting early, mid, and late conceptual states) with classic, strictly regularized machine learning provides the most accurate, interpretable, and computationally efficient solution for detecting LLM hallucinations on small datasets.
