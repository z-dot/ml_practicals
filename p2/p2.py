# %%
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

E = 1e-9


class NBC(BaseEstimator):
    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.is_binary = np.array([t == "b" for t in feature_types])
        self.num_classes = num_classes
        self.features = []
        self.prior = np.ones(num_classes) * E
        self.count = np.zeros(num_classes)
        self.means = np.zeros((num_classes, len(feature_types)))
        self.variances = np.zeros((num_classes, len(feature_types)))

    def fit(self, Xtrain, Ytrain):
        for c_i in range(self.num_classes):
            v = Ytrain == c_i
            self.count[c_i] = sum(v)
            self.prior[c_i] = E + self.count[c_i] / len(Ytrain)
            x_c = Xtrain[v]
            self.means[c_i] = np.mean(x_c, axis=0)  # takes the featurewise mean
            self.variances[c_i] = np.var(x_c, axis=0)

    def predict(self, Xtest):
        assert Xtest.shape[1] == len(self.feature_types)

        n = Xtest.shape[0]

        log_probs = np.zeros((n, self.num_classes))

        for c_i in range(self.num_classes):
            log_prior = np.log(self.prior[c_i])
            binary_means = self.means[c_i, self.is_binary]
            # suppose we have a binary feature j - we wanna calculate
            # the likelihood of x_i,j given that y=c
            # so we take Xtest[i, j].
            binary_features = Xtest[:, self.is_binary]
            binary_likelihood = np.sum(
                binary_features * np.log(binary_means + E)
                + (1 - binary_features) * np.log(1 - binary_means + E),
                axis=1,
            )

            real_features = Xtest[:, ~self.is_binary]
            real_means = self.means[c_i, ~self.is_binary]
            real_variances = self.variances[c_i, ~self.is_binary] + E
            real_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * real_variances)
                + (real_features - real_means) ** 2 / (real_variances),
                axis=1,
            )

            log_probs[:, c_i] = log_prior + binary_likelihood + real_likelihood

        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1, keepdims=True)

        return probs


@dataclass
class Gaussian:
    means: np.ndarray
    variances: np.ndarray


@dataclass
class Bernoulli:
    ps: np.ndarray


# %% [markdown]


# # Handin 1:
# By default, the scikit-learn LR model uses L2 regularisation.
# Assume that we weight all examples equally. Then the penalty term in sklearn is
# 1/2C w^Tw. So we set 0.1=1/2C to get C=5.

# %%

iris = load_iris()

X, y = iris["data"], iris["target"]

# Copied code from practical sheet
N, D = X.shape
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]

# %%


feature_types = ["r", "r", "r", "r"]
num_classes = 3

nbc_accuracies = np.zeros((200, 10))
lr_accuracies = np.zeros((200, 10))

for j in range(200):
    shuffler = np.random.permutation(N)
    Xtrain = X[shuffler[:Ntrain]]
    ytrain = y[shuffler[:Ntrain]]
    Xtest = X[shuffler[Ntrain:]]
    ytest = y[shuffler[Ntrain:]]

    for i in range(1, 11):
        n_samples = int(i * 0.1 * Ntrain)
        Xtrain_subset = Xtrain[:n_samples]
        ytrain_subset = ytrain[:n_samples]

        nbc = NBC(feature_types, num_classes)
        nbc.fit(Xtrain_subset, ytrain_subset)
        nbc_predictions = np.argmax(nbc.predict(Xtest), axis=1)
        nbc_accuracy = np.mean(ytest == nbc_predictions)
        nbc_accuracies[j, i - 1] = nbc_accuracy

        lr = LogisticRegression(C=5, max_iter=200)
        lr.fit(Xtrain_subset, ytrain_subset)
        lr_predictions = lr.predict(Xtest)
        lr_accuracy = np.mean(ytest == lr_predictions)
        lr_accuracies[j, i - 1] = lr_accuracy

nbc_mean_accuracies = np.mean(nbc_accuracies, axis=0)
nbc_std_accuracies = np.std(nbc_accuracies, axis=0)
lr_mean_accuracies = np.mean(lr_accuracies, axis=0)
lr_std_accuracies = np.std(lr_accuracies, axis=0)

plt.errorbar(
    [x - 0.1 for x in range(1, 11)],
    nbc_mean_accuracies,
    yerr=nbc_std_accuracies,
    label="NBC",
    capsize=2,
)
plt.errorbar(
    [x + 0.1 for x in range(1, 11)],
    lr_mean_accuracies,
    yerr=lr_std_accuracies,
    label="LR",
    capsize=2,
)
plt.xticks(range(1, 11), [f"{i*10}%" for i in range(1, 11)])
plt.xlabel("Amount of training data (%)")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %%
