from sklearn.linear_model import Ridge

def sklearn_train_linreg(X, y, alpha):
    return Ridge(alpha=alpha).fit(X, y)