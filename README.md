# Machine Learning Specialization — Stanford / DeepLearning.AI

Completed implementations from Andrew Ng's [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) on Coursera (3 courses, certified Nov–Dec 2025).

## Repository Structure

### Course 1: Supervised Machine Learning
| Notebook | Topics | Key Implementation |
|---|---|---|
| `C1_W2_Linear_Regression` | Gradient descent, cost function | `compute_cost`, `compute_gradient` from scratch |
| `C1_W3_Logistic_Regression` | Sigmoid, binary cross-entropy, regularization | Logistic regression + L2-regularized variant |

### Course 2: Advanced Learning Algorithms
| Notebook | Topics | Key Implementation |
|---|---|---|
| `C2_W1_Neural_Networks_Binary` | TensorFlow Sequential, NumPy forward prop | `my_dense` (single-neuron forward pass) |
| `C2_W2_Neural_Networks_Multiclass` | Softmax, ReLU, 10-class classification | `my_softmax`, multiclass digit recognizer |
| `C2_W3_Bias_Variance_Model_Selection` | Train/CV/test splits, regularization tuning | Complex vs. Simple vs. Regularized NN comparison |
| `C2_W4_Decision_Tree` | Entropy, information gain, recursive splitting | Decision tree built entirely from scratch |

### Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning
| Notebook | Topics | Key Implementation |
|---|---|---|
| `C3_W1_KMeans` | Centroid assignment, image compression | `find_closest_centroids`, `compute_centroids` |
| `C3_W1_Anomaly_Detection` | Gaussian model, F1-based threshold | `estimate_gaussian`, `select_threshold` |
| `C3_W2_Collaborative_Filtering_RecSys` | Custom cost function, GradientTape training | `cofi_cost_func` (loop + vectorized TF), Adam optimizer, L2 regularization |
| `C3_W2_Content_Based_RecSys_NN` | Dual-tower NN, L2-normalized embeddings | User/Item neural networks with dot-product similarity |
| `C3_W3_Deep_Q_Learning_Lunar_Lander` | DQN, experience replay, target network | `compute_loss` (Bellman equation), ε-greedy policy |

## Environment

```
Python 3.10
TensorFlow 2.14.0
NumPy 1.24.3
scikit-learn 1.3.0
matplotlib 3.7.2
```

Originally completed in Coursera Labs (Jupyter). See `requirements.txt` for local reproduction.

**Note:** Notebooks depend on course-provided helper utilities (`utils.py`, data loaders) from the Coursera lab environment. The implementations in each notebook are my own work; infrastructure code is from the course.

## Author

**Aryandeep Singh Gill**
B.E. Electrical & Electronics Engineering, BITS Pilani (8.26/10)
Data Analytics Engineer @ Providence Global Centre
