# %%
import _pickle as cp
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline

# %%
X, y = cp.load(open("winequality-white.pickle", "rb"))

N, D = X.shape
N_train = int(0.8 * N)
N_test = N - N_train

X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]

# %%


print("Handin 1")
unique, counts = np.unique(y, return_counts=True)
plt.bar(unique, counts)
plt.xlabel("Quality")
plt.ylabel("Frequency")
plt.show()

# %%


def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


naive_mse = mse(y, np.mean(y))

print("Handin 2")
print(f"Mean squared error: {naive_mse}")

# %%

X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

X_train_standardized = (X_train - X_train_mean) / X_train_std
X_test_standardized = (X_test - X_train_mean) / X_train_std

print(np.mean(X_train_standardized, axis=0), np.std(X_train_standardized, axis=0))

# %%

# closed form: w = (X^T X)^-1 X^T y

X_train_aug = np.hstack((np.ones((N_train, 1)), X_train_standardized))
X_test_aug = np.hstack((np.ones((N_test, 1)), X_test_standardized))

w = np.linalg.inv(X_train_aug.T @ X_train_aug) @ X_train_aug.T @ y_train

print("Handin 3")

opt_train_loss = mse(y_train, X_train_aug @ w)
opt_test_loss = mse(y_test, X_test_aug @ w)

print(f"train loss: {opt_train_loss}")
print(f"test loss: {opt_test_loss}")

# %%

sizes = list(range(20, 601, 20))
ws = [
    (
        np.linalg.inv(X_train_aug[:i].T @ X_train_aug[:i])
        @ X_train_aug[:i].T
        @ y_train[:i]
    )
    for i in sizes
]
train_loss = [mse(y_train[:i], X_train_aug[:i] @ w) for i, w in zip(sizes, ws)]
test_loss = [mse(y_test, X_test_aug @ w) for w in ws]

print("Handin 4")
plt.plot(sizes, train_loss, label="train")
plt.plot(sizes, test_loss, label="test")
plt.axhline(
    opt_test_loss, color="orange", linestyle="dashed", label="optimal test loss"
)
plt.axhline(naive_mse, color="green", linestyle="dotted", label="mean")
plt.xlabel("size")
plt.ylabel("mse")
plt.ylim(bottom=0)
plt.legend()
plt.show()

print(
    "the plot shows that the test loss is minimized at around 300 samples, with good performance after 200."
)

print(
    "it looks like it's underfitting, given that train and test loss are extremely similar and fairly high"
)
# %%

lambdas = np.logspace(-2, 2, 5, base=10)

y_valid_length = int(0.8 * N_train)
y_valid = y_train[y_valid_length:]
X_valid = X_train[y_valid_length:]
X_train_small = X_train[:y_valid_length]
y_train_small = y_train[:y_valid_length]

ridge_pipeline = sklearn.pipeline.Pipeline(
    [
        sklearn.preprocessing.PolynomialFeatures(2),
        sklearn.preprocessing.StandardScaler(),
        sklearn.linear_model.Ridge(fit_intercept=False),
    ]
)
lasso_pipeline = sklearn.pipeline.Pipeline(
    [
        sklearn.preprocessing.PolynomialFeatures(2),
        sklearn.preprocessing.StandardScaler(),
        sklearn.linear_model.Lasso(fit_intercept=False),
    ]
)


ridges = []
lassos = []


# sklearn's alpha is our lambda, fit_intercept should be false, given that we've already standardised our data

for lamb in lambdas:
    ridge_pipeline.set_params(ridge__alpha=lamb)
    lasso_pipeline.set_params(lasso__alpha=lamb)

    ridge_pipeline.fit(X_train_small, y_train_small)
    lasso_pipeline.fit(X_train_small, y_train_small)

    ridge_score = mse(y_valid, ridge_pipeline.predict(X_valid))
    lasso_score = mse(y_valid, lasso_pipeline.predict(X_valid))
    ridges.append(ridge_score)
    lassos.append(lasso_score)

# best ridge

ridge_lamb = lambdas[np.argmin(ridges)]
ridge_pipeline.set_params(ridge__alpha=ridge_lamb)
ridge_pipeline.fit(X_train, y_train)
ridge_test_loss = mse(y_test, ridge_pipeline.predict(X_test))

# best lasso

lasso_lamb = lambdas[np.argmin(lassos)]
lasso_pipeline.set_params(lasso__alpha=lasso_lamb)
lasso_pipeline.fit(X_train, y_train)
lasso_test_loss = mse(y_test, lasso_pipeline.predict(X_test))

print("ridge: ", ridge_lamb, ridge_test_loss)
print("lasso: ", lasso_lamb, lasso_test_loss)

# %%
