# 🩺 Medical XAI Evaluation

## Comparative Analysis of Explainable AI Approaches for Medical Image Classification Tasks

### Overview

This project investigates and compares multiple Explainable Artificial Intelligence (XAI) techniques for improving the transparency and interpretability of deep learning–based medical image classification models. While machine learning and AI have achieved remarkable success in medical diagnostics, most deep neural networks (DNNs) still function as black boxes, limiting their adoption in healthcare.

To address this, we will implement and evaluate four widely used XAI methods —Grad-CAM, Integrated Gradients, LIME, and SHAP— on a medical image classification dataset. Each method will be quantitatively assessed using a multi-metric evaluation framework to measure how accurately and consistently the explanations reflect model reasoning.

### Objectives

- Develop a quantitative evaluation framework for explainability in medical imaging.
- Compare gradient-based and model-agnostic XAI techniques under the same conditions.
- Measure and analyze interpretability using standardized quantitative metrics.
- Identify the most faithful and robust methods for revealing model decision patterns.

### Explainability Methods

| Method                | Type            | Description                                                                 | Reference                  |
|-----------------------|-----------------|-----------------------------------------------------------------------------|----------------------------|
| Grad-CAM             | Gradient-based | Generates class activation heatmaps highlighting important image regions.   | Selvaraju et al., 2017    |
| Integrated Gradients | Gradient-based | Calculates pixel-wise attributions by integrating gradients from a baseline to input. | Sundararajan et al., 2017 |
| LIME                 | Model-agnostic | Creates local surrogate models over superpixels to approximate local decision boundaries. | Ribeiro et al., 2016      |
| SHAP                 | Model-agnostic | Assigns feature importance using Shapley values from cooperative game theory. | Lundberg & Lee, 2017      |

### Evaluation Metrics

| Metric                | Purpose            | Description                                                                 |
|-----------------------|-------------------|-----------------------------------------------------------------------------|
| Faithfulness / Fidelity | Causal correctness | Measures confidence drop when salient pixels are removed (Deletion–Insertion AUC). |
| Localization Accuracy | Spatial relevance | Computes overlap between explanation maps and lesion regions (IoU, Dice).   |
| Robustness / Stability | Reliability        | Evaluates consistency of explanations under input perturbations (SSIM, correlation). |
| Infidelity / Sensitivity-n | Causal reliability | Tests how closely explanation values match model output changes under perturbations. |

### Dataset

The experiments will use publicly available medical-imaging datasets such as:

- **Brain MRI Dataset** – for tumor vs. non-tumor classification

### Project Workflow

1. **Model Training** – Train a foundation model on medical-imaging data for binary classification.
2. **Explanation Generation** – Apply Grad-CAM, Integrated Gradients, LIME, and SHAP.
3. **Quantitative Evaluation** – Compute four metrics (Faithfulness, Localization, Robustness, Infidelity).
4. **Comparison & Visualization** – Aggregate and visualize metric results to identify the most reliable method.

### References

- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV 2017.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. ICML 2017.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?” Explaining the predictions of any classifier. KDD 2016.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS 2017.
- Samek, W., et al. (2017). Evaluating the visualization of what a deep neural network has learned. IEEE TNNLS 28(11).
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence 2(11).
- DeGrave, A. J., Janizek, J. D., & Lee, S. I. (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence 3(7).