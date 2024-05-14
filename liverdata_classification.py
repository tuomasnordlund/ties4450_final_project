# -*- coding: utf-8 -*-

# %% ==========================================================================
""" 
Reading the data
"""
import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/Kone/Documents/harrasteluastna/liver/Indian Liver Patient Dataset (ILPD).csv", na_values=np.nan)
rng = np.random.RandomState(19583563)

# %% ==========================================================================
"""
Data exploration
"""
# =============================================================================
# Age       : Age of patient
# Gender    : Gender of patient
# TB        : Total Bilirubin level (high = bad)
# DB        : Direct Bilirubin level (high = bad)
# Alkphos   : Alkaline Phosphatase level (high = bad)
# Sgpt      : SGPT level (high = bad)
# Sgot      : SGOT level (high = bad)
# TP        : Total proteins
# ALB       : Albumin level
# A/G ratio : Ratio of albumin and globulin (1.1 - 2.5 normal; 3.4-5.4 for A, 2.0-3.5 for G)
# Selector  : Healthy / ill
# =============================================================================

print(df.shape) # data dimensions: 583 rows, 11 columns
#len(df.index)
print(df.columns)
[print(df[name].describe()) for name in df.columns]
print(df.head())
print(df.describe())
print(df.dtypes)
# Gender data type is object
print(df['Gender'].head())

#%%

print([df[name].describe() for name in df.columns])

#%%

# Check for null values
print(df.isnull().any())
print(df.isnull().sum())
null_mask = df.isnull().any(axis=1) # Create a vector of Boolean values
print(len(df[null_mask]))
print(df[null_mask])
print(df['A/G Ratio'].mean())
# We could replace the null values with the A/G Ratio mean, but let's leave it as is for now

print(df['Selector'].value_counts()) # Selector value 1 = healthy, 2 = diseased

# %% 
# Compute class means for A/G Ratio in case we want to replace null values with them
# Other option is to just use the A/G Ratio mean

#agratio_mean_healthy = df[df['Selector'] == 1].mean()['A/G Ratio']
#agratio_mean_ill = df[df['Selector'] == 2].mean()['A/G Ratio']

#%%

print(df.skew().sort_values(ascending=False))

# %% ==========================================================================
"""
Data visualization

Dimensions and the amount of subplots are hard-coded for simplification.
This is bad practice.
"""

# Plotting columns as histograms will fail if gender is not a numeric value:
df.replace(['Female', 'Male'], [0,1], inplace=True)
df.head()


# %%
import matplotlib.pyplot as plt

def plot_histograms(df):
    # 10 subplots are divided to two plots to deal with the subplots overlapping
    fig, axs = plt.subplots(2, 2)
    for i, column in enumerate(df.columns[0:4]):
        if column == 'Gender':
            df[column].value_counts().plot(kind='bar', ax=axs.ravel()[i], title=column)
        else:
            df[column].plot(kind='hist', ax=axs.ravel()[i], title=column)
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(3, 2)
    for i, column in enumerate(df.columns[4:10]):
        df[column].plot(kind='hist', ax=axs.ravel()[i], title=column)
    plt.tight_layout()
    plt.show()

plot_histograms(df)

#%%
# Create boxplots in a similar fashion

def plot_boxplots(df):
    fig, axs = plt.subplots(2, 2)
    for i, column in enumerate(df.columns[0:4]):
        if column == 'Gender':
            continue
        else:
            df[column].plot(kind='box', ax=axs.ravel()[i], title=column)
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(3, 2)
    for i, column in enumerate(df.columns[4:10]):
        df[column].plot(kind='box', ax=axs.ravel()[i], title=column)
    plt.tight_layout()
    plt.show()

plot_boxplots(df)

#%%
# Plot values based on the dependent variable.
# Hide outliers to make the plots readable.
import seaborn as sns

fig, axs = plt.subplots(2, 2)
for i, column in enumerate(df.columns[2:6]):
    sns.boxplot(x='Selector', y=column, ax=axs.ravel()[i], data=df, showfliers=False)
fig.tight_layout()
plt.show()
fig, axs = plt.subplots(2, 2)
for i, column in enumerate(df.columns[6:10]):
    sns.boxplot(x='Selector', y=column, ax=axs.ravel()[i], data=df, showfliers=False)
fig.tight_layout()
plt.show()


#%%
n = len(df.columns)-1
X = df.iloc[:, 0:n]
y = df['Selector']
y = y-1


#%%
"""
Replace NaN values with KNN Imputer
"""

#from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(weights='distance')
X = imputer.fit_transform(X)
X_feature_names = imputer.get_feature_names_out()
print(X[null_mask])

X = pd.DataFrame(X, columns=X_feature_names)

#%%
""" 
Assess correlations between variables
"""

from scipy.stats import spearmanr

X_transpose = X.values.T
X_corr = np.corrcoef(X_transpose) # Pearson correlation coefficient
print(X_corr)

X_corr_spearman, X_spearman_p = spearmanr(X, axis=0)
print(X_spearman_p)

#%%
"""
Correlation between all the variables visualized
"""

fig, axs = plt.subplots(4, 4)
for i in range(2, 6):
    for j in range(2, 6):
        axs[i-2,j-2].scatter(df[df.columns[i]], df[df.columns[j]], c=df['Selector'], marker=".")
fig.tight_layout()
plt.show()
fig, axs = plt.subplots(4, 4)
for i in range(6, 10):
    for j in range(6, 10):
        axs[i-6,j-6].scatter(df[df.columns[i]], df[df.columns[j]], c=df['Selector'], marker=".")
fig.tight_layout()
plt.show()

print(df[df.columns[2:10]].corr())
sns.heatmap(df[df.columns[2:10]].corr())

#%%
"""
Get VIF scores for multicollinearity
"""
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = pd.DataFrame(X, columns=X_feature_names)
X_vif['intercept'] = 1
vif_data = pd.DataFrame()
#add_constant(X_vif)
vif_data['Feature'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data)

#%%
"""
Split data into training and testing sets
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rng)


#%%
"""
Test how well a logistic regression model captures the raw data
"""

clf = LogisticRegression(solver='newton-cholesky', random_state=rng)
clf.fit(X_train, y_train)

#%% Get coefficients

print(X_feature_names)
print(clf.intercept_)
print(["{:.5f}".format(float(val)) for val in clf.coef_[0]])

#%% Odds ratios
print(np.exp(clf.intercept_))
print(np.exp(clf.coef_))

#%% Prediction and test scores
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print(clf.predict(X_test[:5, :]))
print(y_test[0:5])
print(clf.score(X_test, y_test))
clf_pred = clf.predict(X_test)
print(mean_squared_error(y_test, clf_pred))
#print(accuracy_score(y_test, clf_pred))

clf_confmatrix = confusion_matrix(y_test, clf_pred, labels=[0,1])

#%%
# Visualize odds ratios
logreg_or = pd.Series(
    np.exp(clf.coef_[0]),
    index=X_feature_names
    ).sort_values(ascending=True)

logreg_or.plot(kind='barh')

#%%
"""
Random Forest is robust against scale differences in the data.
It can also be used to extract most important features.
"""
from sklearn.ensemble import RandomForestClassifier
clf_forest = RandomForestClassifier(random_state=rng)
clf_forest.fit(X_train, y_train)

#%% Get prediction accuracy

print(clf_forest.score(X_test, y_test))
clf_rf_confmatrix = confusion_matrix(y_test, clf_forest.predict(X_test), labels=[0,1])

#%%

print(X_feature_names)
print(clf_forest.feature_importances_)

rf_importances = pd.Series(
    clf_forest.feature_importances_, 
    index=X_feature_names
    ).sort_values(ascending=True)

rf_importances.plot(kind='barh')


#%%
"""
Test if KNN can reliably classify the data.
"""

from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(n_neighbors=2)
clf_knn.fit(X_train, y_train)

#%%

print(clf_knn.score(X_test, y_test))
clf_knn_confmatrix = confusion_matrix(y_test, clf_knn.predict(X_test), labels=[0,1])

#%%
"""
Test how XGBoost Classifier can deal with the data.
"""


from xgboost import XGBClassifier
from sklearn.metrics import classification_report

clf_xgb = XGBClassifier(n_estimators=10, random_state=rng)
clf_xgb.fit(X_train, y_train)
xgb_pred = clf_xgb.predict(X_test)
print(accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

#%%
"""
With logistic regression, random forest, KNN and XGBoost classifiers without
modifications or preprocessing we reach quite good classification scores.

We can try improving the score by processing the data for example with 
scalarization and PCA.

We will start by attempting to visualize the two most important components.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
pca_2c = PCA(n_components=2, random_state=rng)

X_train_scaled = scaler.fit_transform(X_train)
X_pca = pca_2c.fit_transform(X_train_scaled)

#%%

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train)
print(pca_2c.components_)
print(pca_2c.explained_variance_)

#%%
"""
Test how explained variance increases with the number of principal components.
"""

n_comps = np.arange(10)
var_ratios = []
for n in n_comps:
    pca = PCA(n_components=n)
    pca.fit(X_train_scaled)
    var_ratios.append(np.sum(pca.explained_variance_ratio_))

plt.grid()
plt.plot(n_comps,var_ratios,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.title('n_components vs. Explained Variance Ratio')
plt.show()

#%%
"""
We can now test how different scalers and PCA affect the model accuracy.
Let's define a function for this.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer

def test_models(X_train, X_test, y_train, y_test, is_pca: bool, scale=None):
    pca = PCA(n_components=0.8, random_state=rng)
    if scale == 'robust':
        scaler = RobustScaler()
    elif scale == 'scalarize':
        scaler = StandardScaler()
    elif scale == 'yeo-johnson':
        scaler = PowerTransformer(method='yeo-johnson') 

    models = [
        LogisticRegression(solver='newton-cholesky', max_iter=300, random_state=rng),
        RandomForestClassifier(random_state=rng),
        KNeighborsClassifier(n_neighbors=2),
        XGBClassifier(random_state=rng)
        ]

    names = [
        'Logistic Regression',
        'Random Forest Classification',
        'KNN',
        'XGB Classifier'
        ]

    results = {}
    for name, model in zip(names, models):
        if (scale and is_pca):
            pipe = Pipeline([ ('scaler', scaler), ('PCA', pca), (name, model) ])
        elif scale:
            pipe = Pipeline([ ('scaler', scaler), (name, model) ])
        elif is_pca:
            pipe = Pipeline([('PCA', pca), (name, model) ])
        else:
            pipe = Pipeline([(name, model)])
        pipe.fit(X_train, y_train)
        results.update({name: round(pipe.score(X_test, y_test), 4)})
    
    return results

#%%

def run_tests(X_train, X_test, y_train, y_test):
    scalers = ['robust', 'scalarize', 'yeo-johnson']
    results_raw = test_models(X_train, X_test, y_train, y_test, is_pca=False)
    results_no_scaler = test_models(X_train, X_test, y_train, y_test, is_pca=True)
    res = {'Model only': results_raw,
           'pca': results_no_scaler}
    
    for s in scalers:
        results = test_models(X_train, X_test, y_train, y_test, scale=s, is_pca=True)
        results_no_pca = test_models(X_train, X_test, y_train, y_test, scale=s, is_pca=False)
        res.update(
            {s: {
                'scaler + pca': results, 
                'scaler': results_no_pca
                }
            })
    #print(results)
    #print(results_no_scaler)
    #print(results_no_pca)
    #print(results_raw)
    return res
    
res = run_tests(X_train, X_test, y_train, y_test)

#%%
"""
Remove collinear variables based on VIF score.
"""

def remove_collinear(X):
    # X_vif = X.copy()
    X_vif = pd.DataFrame(X, columns=X_feature_names)
    X_vif['intercept'] = 1
    dropped_features = []
    
    collinear_features = True
    while collinear_features:
        vif_data = pd.DataFrame()
        vif_data['Feature'] = X_vif.columns
        
        vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        vif_data = vif_data[vif_data['Feature']!='intercept']
        
        if (vif_data['VIF'] > 5).any():
            max_vif_idx = np.argmax([vif_data['VIF']])
            feature_to_remove = vif_data['Feature'][max_vif_idx]
            dropped_features.append(feature_to_remove)
            X_vif.drop(feature_to_remove, axis='columns', inplace=True)
        else:
            collinear_features = False
    X_vif = X_vif.drop('intercept', axis='columns')
    return X_vif, dropped_features

X_train_vif, removed_features = remove_collinear(X_train)
X_test_vif = X_test.drop(removed_features, axis='columns')
sns.heatmap(X_train_vif[2:].corr())

res_vif = run_tests(X_train_vif, X_test_vif, y_train, y_test)

#%%
"""
Drop outliers using the LocalOutlierFactor.
"""
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor()
y_lof = lof.fit_predict(X_train)
mask = y_lof != -1
mask2 = y_lof == -1
X_train_lof = X_train[mask]
y_train_lof = y_train[mask]
X_outliers = X_train[mask2]

plot_histograms(X_train)
plot_histograms(X_train_lof)

#%%

res_lof = run_tests(X_train_lof, X_test, y_train_lof, y_test)


#%%
"""
Drop outliers based on IQR ranges.
"""

def remove_outliers(data):
    cdata = data.copy()
    # Save outliers in a dataframe
    outl = pd.DataFrame()
    
    # Search for outliers in all columns except age and gender
    for var in cdata.columns[2:]:
        q1 = cdata[var].quantile(0.25)
        q3 = cdata[var].quantile(0.75)
        IQR = q3 - q1
        # We will use a k of 3 to remove very extreme values
        lb = q1 - 3 * IQR       # lower bound
        ub = q3 + 3 * IQR       # upper bound
        outliers = cdata[(cdata[var] < lb) | (cdata[var] > ub)]
        outl = pd.concat([outl, outliers])
        cdata = cdata.drop(outliers.index, axis='index')
        
    return cdata, outl

X_train_iqr, X_iqr_outliers = remove_outliers(X_train)
y_train_iqr = y_train.drop(X_iqr_outliers.index)
plot_histograms(X_train_iqr)
#%%

res_iqr = run_tests(X_train_iqr, X_test, y_train_iqr, y_test)

#%%
"""
Combine IQR with VIF to create a dataset with less outliers and collinearity
"""

X_train_iqr_vif, X_vif_outliers = remove_outliers(X_train_vif)
y_train_iqr_vif = y_train.drop(X_vif_outliers.index)
plot_histograms(X_train_iqr_vif)

#%%

res_iqr_vif = run_tests(X_train_iqr_vif, X_test_vif, y_train_iqr_vif, y_test)

#%%
"""
Recursively find keys associated with the maximum value in a nested dictionary.
"""

def max_value_keys(d):    
    max_val = float('-inf')
    max_keys = []

    for k, v in d.items():
        if isinstance(v, dict):
            val, nested_keys = max_value_keys(v)
            if nested_keys:
                if val > max_val:
                    max_val = val
                    max_keys = [k] + nested_keys
        else:
            if v > max_val:
                max_val = v
                max_keys = [k]

    return max_val, max_keys

#%%
res_best, res_keys = max_value_keys(res)
res_vif_best, res_vif_keys = max_value_keys(res_vif)
res_lof_best, res_lof_keys = max_value_keys(res_lof)
res_iqr_best, res_iqr_keys = max_value_keys(res_iqr)
res_iqr_vif_best, res_iqr_vif_keys = max_value_keys(res_iqr_vif)

print(f'Best score without preprocessing: {res_best} found with {res_keys}')
print(f'Best score with VIF: {res_vif_best} found with {res_vif_keys}')
print(f'Best score with LOF: {res_lof_best} found with {res_lof_keys}')
print(f'Best score with IQR: {res_iqr_best} found with {res_iqr_keys}')
print(f'Best score with IQR and VIF: {res_iqr_vif_best} found with {res_iqr_vif_keys}')

#%%

"""
There are no noteworthy improvements above a score ~0.76. In general the best results
are reached with logistic regression and random forest. For now 
we will choose Random Forest as our model and tune its hyperparameters.
Based on the scores the best combination is either to use robust scaler 
or VIF and Yeo-Johnson.
"""
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(100, 500, num=10)]
max_depth = [[None], [int(x) for x in np.linspace(1, 15, 15)]]
max_depth = [x for xs in max_depth for x in xs]
min_samples_split = np.arange(2,10)
max_features = ['sqrt', 'log2']
bootstrap = [True, False]

rf = RandomForestClassifier()
param_dist = {'model__n_estimators': n_estimators,
              'model__max_depth': max_depth,
              'model__min_samples_split': min_samples_split,
              'model__max_features': max_features,
              'model__bootstrap': bootstrap}

pipe_rf = Pipeline(steps=[('scaler', PowerTransformer()),
                          ('model', rf)])

rs = RandomizedSearchCV(pipe_rf,
                        param_dist,
                        n_iter=50,
                        cv=5,
                        random_state=rng)

rs2 = RandomizedSearchCV(pipe_rf,
                        param_dist,
                        n_iter=50,
                        cv=5,
                        random_state=rng)

rs.fit(X_train_vif, y_train)
rs2.fit(X_train, y_train)
#%%

print(rs.best_params_)
print(rs.best_score_)
print(rs2.best_params_)
print(rs2.best_score_)

#%%

pipe_rf2 = Pipeline(steps=[('scaler', RobustScaler()),
                           ('model', rf)])

rs3 = RandomizedSearchCV(pipe_rf2,
                        param_dist,
                        n_iter=50,
                        cv=5,
                        random_state=rng)

rs3.fit(X_train, y_train)

print(rs3.best_params_)
print(rs3.best_score_)

rs3.fit(X_train_vif, y_train)

print(rs3.best_params_)
print(rs3.best_score_)


#%%
"""
We can try optimizing linear regression as well
"""

n_comp = np.arange(2,11)
whiten = [True, False]

penalty = [None, 'l1', 'l2']
C = np.linspace(1, 5)
solver = ['liblinear', 'saga']

pca = PCA()
lr = LogisticRegression()

param_dist = {'pca__n_components': n_comp,
              'pca__whiten': whiten,
              'model__penalty': penalty,
              'model__C': C,
              'model__solver': solver}

pipe_logreg = Pipeline(steps=[('scaler', StandardScaler()),
                              ('pca', pca),
                              ('model', lr)])

rs_lr = RandomizedSearchCV(pipe_logreg,
                           param_dist,
                           n_iter=50,
                           cv=5,
                           random_state=rng)

rs_lr2 = RandomizedSearchCV(pipe_logreg,
                           param_dist,
                           n_iter=50,
                           cv=5,
                           random_state=rng)

rs_lr.fit(X_train, y_train)
rs_lr2.fit(X_train_iqr, y_train_iqr)

#%%

print(rs_lr.best_params_)
print(rs_lr.refit_time_)
print(rs_lr.best_score_)
print(rs_lr2.best_params_)
print(rs_lr2.refit_time_)
print(rs_lr2.best_score_)



#%%

















