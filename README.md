# Author
Alexis Marion, CEA List <br>
<img src="https://avatars1.githubusercontent.com/u/24958773?s=200&v=4" alt="drawing" width="80"/>

# Partenarial Explainer

Partenarial Explainer is a method of interpretability based on the concept of partenarial examples. For a binary classification task, the method aims, for a selected input, to find the closest example in the other class. In a fault detection task this method helps to identify the actions to take to 'repair' a faulty example.

This method is applied to XGBoost models. The first step consist in approximating the XGBoost model with a differentiable one with a method called DFE (Differentiable Forest Estimator). Then we research a partenarial example with DDN (Decoupling Direction and Norm[1]).

# Demo

demo.ipynb give you a quick example and results on 3 distincts datasets.

# References

[1] Rony, J., Hafemann, L.G., Oliveira, L.S., Ayed, I.B., Sabourin, R., Granger, E., 2018. Decoupling Direction and Norm for Efficient Gradient-Based L2 Adversarial Attacks and Defenses. arXiv:1811.09600 [cs].
