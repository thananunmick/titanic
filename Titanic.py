import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_auc_score

# load training dataset and return it as a pandas dataframe
def load_train_set():
    train_path = os.path.join("dataset", "train.csv")
    return pd.read_csv(train_path)

train = load_train_set()
# print(train.head())

train_drop = train.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
train_drop_na = train_drop.dropna()
X_train = train_drop_na.drop("Survived", axis=1)
y_train = train_drop_na["Survived"].copy()
# train_dropna = train.dropna()
# X_train = train_dropna.drop(["Survived", "Name", "Ticket", "Cabin", "PassengerId"], axis=1)
# X_train_nm = X_train.drop(["Sex", "Embarked"], axis=1)
# y_train = train_dropna["Survived"].copy()

# Use SimpleImputer to fill in any nan value to the mean of that column
# def fill_in_nan():
#     imputer = SimpleImputer(strategy="mean")
#     return imputer.fit_transform(X_train)

# X_train = fill_in_nan()

# Using OrdinalEncoder to change text features to number
X_train_cat = X_train[["Sex", "Embarked"]].copy()
X_train_nm = X_train.drop(["Sex", "Embarked"], axis=1)
oe = OrdinalEncoder()
X_train_encoded = oe.fit_transform(X_train_cat)

# Turn X_train_encoded back to dataframe and join it with normal X_train
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=X_train_cat.columns, index=X_train_cat.index)
X_train_final = X_train_nm.join(X_train_encoded_df)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_final)

# Choose KNeighborsClassifier as the model to train the training dataset
def run_kneighbor_clf():
    grid_param = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'weights': ['uniform', 'distance']}
    knn_clf = KNeighborsClassifier()
    grid = GridSearchCV(knn_clf, grid_param)
    return grid.fit(X_train_final, y_train)

def run_svc_clf():
    grid_param = {'kernel': ['poly', 'rbf', 'sigmoid', 'linear'], 'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    svc_clf = SVC()
    grid = GridSearchCV(svc_clf, grid_param, cv=3, scoring="accuracy")
    return grid.fit(X_train_final, y_train)

def run_log_reg_clf():
    log_reg = LogisticRegression(penalty='l2', C=50)
    return log_reg.fit(X_train_final, y_train)

# Main train function
def train():
    # model = run_kneighbor_clf()
    model = run_svc_clf()
    # model = run_log_reg_clf()
    return model

model = train()

print("Done Training")
print(model.best_estimator_)

# print(y_train.value_counts())

# Evaluation
def evaluate():
    # Use CrossValScore to test the accruacy of the model
    print(cross_val_score(model, X_train_final, y_train, cv=3, scoring="accuracy"))

    y_pred = cross_val_predict(model, X_train_final, y_train, cv=3)
    # Printing Precision and Recall score
    print("Precision score: ", precision_score(y_train, y_pred))
    print("Recall score: ", recall_score(y_train, y_pred))
    precision, recall, thresholds = precision_recall_curve(y_train, y_pred)
    plt.plot(recall, precision, "b-")
    plt.ylabel("Precision")
    plt.xlabel("Recall")

    # Compute and print ROC AUC score
    roc_auc = roc_auc_score(y_train, y_pred)
    print("ROC_AUC score: ", roc_auc)

evaluate()

X_train_predict = X_train_final[19]
print(model.predict([X_train_predict]))

plt.show()

###############################################################
#Test set predictions
def run_test_set():
    def load_test_set():
        test_path = os.path.join("dataset", "test.csv")
        return pd.read_csv(test_path)

    test = load_test_set()
    test_pass_id = test[["PassengerId"]].copy().to_numpy()
    test_nm = test.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
    test_nm = test_nm.fillna(test_nm["Age"].mean())
    test_cat = test[['Sex', 'Embarked']].copy()
    test_nm = test_nm.drop(['Sex', 'Embarked'], axis=1)
    test_cat_encoded = oe.fit_transform(test_cat)
    test_cat_encoded_df = pd.DataFrame(test_cat_encoded, columns=test_cat.columns, index=test_cat.index)
    # print(len(test_nm), len(test_cat_encoded_df))
    # print(test_nm.head(), test_cat_encoded_df.head())
    test_final = test_nm.join(test_cat_encoded_df)
    test_final = scaler.fit_transform(test_final)

    print(test_nm['Age'].isnull().values.any())

    # prediction = cross_val_predict(grid, test_final, cv=3)
    # print(prediction)
    test_final_first = test_final[0]
    prediction = model.predict(test_final)
    prediction = prediction.reshape(418, 1)
    prediction = np.c_[test_pass_id, prediction]
    prediction = prediction.astype(np.int32)
    # np.savetxt("Prediction.csv", prediction, delimiter=",", fmt="%i")
    # print(prediction)
    # print(test_final)