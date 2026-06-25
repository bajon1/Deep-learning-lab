# Lab 10: XAI (Explainable AI)

---

## Task 1: Permutation Feature Importance on a toy dataset

#### Goal: Evaluate feature importance of a multiclass classifier using permutation feature importance.

1. Generate a classification toy dataset using `make_classification` (5000 samples, 20 features, 4 classes, 4 informative features). Convert to `float32`/`long` tensors and split into train/val and test subsets.
2. Train a neural network classifier using cross-validation (5 folds). Save the best model checkpoint per fold — this produces an ensemble of 5 models.
3. Implement a scikit-learn-compatible wrapper for the PyTorch classifier (as in Task 3 of Lab 9).
4. Evaluate feature importance for each ensemble model using `permutation_importance` from `sklearn.inspection` (`n_repeats=100`). Select features significant at the 5% level: a feature is considered significant if its 95% confidence interval (across `n_repeats`) does not contain zero. Identify any features with negative importance values and explain their effect on model performance.
5. Visualize the distribution of feature importance weights across all features.

**Assignment:** Evaluate feature importance using permutation feature importance on the toy classification dataset.

---

## Task 2: Partial Dependence Plots (PDP) on a toy dataset

#### Goal: Estimate the marginal effect of individual features on the output of a regression model.

The partial dependence of model $f$ on feature $X_S$ is defined as:

$$pd_{X_S}(x_S) \overset{\text{def}}{=} \mathbb{E}_{X_C}\left[f(x_S, X_C)\right] = \int f(x_S, x_C)\, p(x_C)\, dx_C$$

where $X_C$ denotes all features except $X_S$. A **Partial Dependence Plot (PDP)** is the plot of $pd_{X_S}(x_S)$ against $x_S$. An **Individual Conditional Expectation (ICE)** plot shows $f(x_S, x_C^{(i)})$ for individual samples.

1. Generate a regression toy dataset using `make_regression` (5000 samples, 20 features). Convert to `float32` tensors and split into train/val and test subsets.
2. Train a neural network regressor using cross-validation (5 folds). Save the best model checkpoint per fold.
3. Implement a scikit-learn-compatible wrapper for the PyTorch regressor.
4. Evaluate feature importance using permutation feature importance on the regression model.
5. Compute and visualize PDPs for the most important features using `PartialDependenceDisplay.from_estimator` from `sklearn.inspection`. Plot: individual feature PDPs, a 2D interaction PDP for a pair of features, and a combined PDP + ICE plot (`kind='both'`).

**Assignment:** Assess the type of relationship between the target variable and the most important features using PDP.

---

## Task 3: LIME on a toy dataset

#### Goal: Evaluate feature importance for individual predictions using the LIME method.

1. Using the regression data and model from Task 2, generate a local neighbourhood around a chosen test sample `X_test[0]` by adding Gaussian perturbations scaled by per-feature standard deviations (`pert_factor=0.3`, `num_of_perturbations=1000`). Compute model outputs and sample-to-perturbation distances:
$$d_i = \exp\left(-\|x_{\text{perturbed}}^{(i)} - x_{\text{explained}}\|\right)$$
2. Fit a weighted least-squares linear model to the perturbations and their outputs using `statsmodels`, with the distances as sample weights.
3. Assess the quality of the fitted linear surrogate model using its $R^2$ value.
4. Evaluate feature importance for the explained sample based on the $t$-statistics of the linear model coefficients.
5. Repeat steps 2–4 for all samples in the test set. Compute mean feature importances across the test set and compare them with the permutation feature importance results from Task 2.

**Assignment:** Evaluate feature importance using LIME on the toy dataset and compare with permutation feature importance.

---

## Task 4: Shapley Values on a toy dataset I

#### Goal: Evaluate feature importance using a sampling-based approximation of Shapley values.

1. Using the regression data and model from Task 2, generate $K$ random subsets of feature indices by sampling subset sizes from $\text{Binomial}(N, 0.5)$ and selecting features at random. Build binary masks from these subsets — a value of $0$ in the mask means the corresponding feature of `X_test[0]` is replaced by its training mean.
2. For each feature $i$, compute its approximate Shapley value as the mean difference in model output when feature $i$ is included versus excluded from the masked input:
$$\varphi(i) \approx \frac{1}{K} \sum_{k=1}^K \left[\nu(S_k \cup \{i\}) - \nu(S_k)\right]$$
3. Evaluate feature importances based on the computed Shapley values for the single explained sample.
4. Repeat steps 2–3 over all test samples with a sufficiently large $K$. Compute mean Shapley values across the test set and compare with permutation feature importance (Task 2) and LIME (Task 3).

**Assignment:** Evaluate feature importance using SHAP on the toy dataset and compare with permutation feature importance and LIME.

---

## Task 5: Shapley Values on a toy dataset II

#### Goal: Implement the exact definitional formula for Shapley values.

The Shapley value of feature $i$ is defined as:

$$\varphi(i) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N| - |S| - 1)!}{|N|!} \left(\nu(S \cup \{i\}) - \nu(S)\right)$$

where $S$ ranges over all subsets of $N \setminus \{i\}$ (all features except $i$), and $\nu(S)$ denotes the model output with features outside $S$ replaced by their training means.

The sampling-based implementation from Task 4 has three limitations: (1) subsets are sampled from the full set $N$ rather than $N \setminus \{i\}$; (2) the sampling does not guarantee unique subsets; (3) for small $N$ (e.g. $\leq 10$), all subsets of $N \setminus \{i\}$ can be enumerated exactly.

1. Implement the exact Shapley value computation by enumerating all subsets of $N \setminus \{i\}$ for each feature $i$, free of the limitations listed above.
2. Evaluate feature importances based on the exact Shapley values for the data from Task 4.
3. Compare the exact Shapley values with the approximate values from Task 4.

**Assignment:** Implement the exact definitional method for computing Shapley values.

---

## Task 6: XAI on a real-world dataset

#### Goal: Explain a black-box model trained on a real-world dataset.

1. Select a real-world classification or regression dataset (e.g. from OpenML, `ucimlrepo`, or scikit-learn). Preprocess features and targets, convert to tensors, and split into train/val and test subsets.
2. Train a neural network model using cross-validation. Save the best model checkpoint per fold.
3. Apply the XAI methods from Tasks 1–4: permutation feature importance, PDP (for regression), LIME, and Shapley values.
4. Compare feature importance rankings across all methods and interpret the results in the context of the dataset.

**Assignment:** Train and explain a classifier/regressor on a real-world dataset using the XAI methods from Tasks 1–4.