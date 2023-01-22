# Tabular Classification with an Employee Attrition Dataset

## <mark style="background: #FFB86CA6;">My Goal of doing this comp</mark> 

> the goals of the Playground Series remain the same—to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. 

- I want to use this comp to build a <mark style="background: #D2B3FFA6;">template</mark> to  <mark style="background: #BBFABBA6;">try as many models and techniques as I could</mark> and <mark style="background: #BBFABBA6;">iterate as fast as I can</mark> 
- This <mark style="background: #D2B3FFA6;">template</mark> should show me<mark style="background: #BBFABBA6;"> all the steps</mark> to tackle a comp from start to finish, from surface scratch to fairly deep
- <mark style="background: #BBFABBA6;">Try as many models as I can</mark> by learning from notebooks and discussions shared on Kaggle





#### <mark style="background: #FFB86CA6;">What about generated dataset</mark> 

> These will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

> Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. 

> While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have <mark style="background: #BBFABBA6;">far fewer artifacts</mark> . 

What are `artifacts` and why they sound like a bad thing? #question 


---

#### <mark style="background: #FFB86CA6;">Overview of the Dataset and the problem</mark> 

##### <mark style="background: #FFB8EBA6;">Key stats</mark> 

- Train: (1677, 35)
- Test: (1119, 34)
- numeric columns: 25 (excluding `id` and `Attrition`)
- categorical columns: 8
- total feature columns: 33
- target column: 1
- columns to drop: 1, `id`

##### <mark style="background: #FFB8EBA6;">files of the official dataset</mark> [dataset](https://www.kaggle.com/competitions/playground-series-s3e3/data?select=train.csv) 

-   **train.csv** - the training dataset; `Attrition` is the binary target
-   **test.csv** - the test dataset; your objective is to predict the probability of positive `Attrition`
-   **sample_submission.csv** - a sample submission file in the correct format

##### <mark style="background: #FFB8EBA6;">What to predict</mark> 

> your objective is to predict the probability of positive `Attrition` (using all other columns)

##### <mark style="background: #FFB8EBA6;">What the submission format</mark> 

For each `EmployeeNumber` in the test set, you must predict the probability for the target variable `Attrition`. The file should contain a header and have the following format:

```
EmployeeNumber,Attrition
1677,0.78
1678,0.34
1679,0.55
etc.
```




---


## <mark style="background: #FFB86CA6;">Baseline and milestone notebooks </mark> 

#### <mark style="background: #FFB8EBA6;">Radek provides a wonderful baseline</mark>  
Radek [notebook](https://www.kaggle.com/code/radek1/eda-training-a-1st-model-submission), Daniel's polars [implementation](https://www.kaggle.com/code/danielliao/build-up-from-radek)
- generated and original dataset joined 
- data source label
- categorical column encoded 
- stratified KF 
- LightGBM with specified categorical columns
- default catboost  
- ensemble the predictions from 20 models

---

## <mark style="background: #FFB86CA6;">The Template (things to do) to build pipelines</mark> 


##### <mark style="background: #FFB8EBA6;">How to read the dataset files</mark> 

access dataset  [[Playground Series Season 3, Episode 3#^146e83|codes]] 
check out the dataset from Kaggle [site](https://www.kaggle.com/competitions/playground-series-s3e3) 

##### <mark style="background: #FFB8EBA6;">How many Feature columns</mark> : 33 
Excluding `id` and `Attrition`

> `Age`, `BusinessTravel`, `DailyRate`, `Department`, `DistanceFromHome`, `Education`, `EducationField`, `EmployeeCount`, `EnvironmentSatisfaction`, `Gender`, `HourlyRate`, `JobInvolvement`, `JobLevel`, `JobRole`, `JobSatisfaction`, `MaritalStatus`, `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `Over18`, `OverTime`, `PercentSalaryHike`, `PerformanceRating`, `RelationshipSatisfaction`, `StandardHours`, `StockOptionLevel`, `TotalWorkingYears`, `TrainingTimesLastYear`, `WorkLifeBalance`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`.

##### <mark style="background: #FFB8EBA6;">How many Categorical columns: 8</mark> 
> `'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime'` 

- How to find out the dtypes of each column?  [[Playground Series Season 3, Episode 3#^d02f8d|codes]] 
- How to find out which columns are Utf8/String or numeric or Int64? [[Playground Series Season 3, Episode 3#^d02f8d|codes]] 

##### <mark style="background: #FFB8EBA6;">How many Numerical columns: 27</mark> (including `id` and `Attrition`)

- How to find out which columns are numeric or Int64? [[Playground Series Season 3, Episode 3#^d02f8d|codes]] 

##### <mark style="background: #FFB8EBA6;">How many Target column: 1</mark> , `Attrition`



##### <mark style="background: #FFB8EBA6;">How many column to ignore: 2</mark> ,  `id` and `EmployeeNumber`

How do I know which columns to ignore?
- `id` from `train_generated` and `EmployeeNumber` from `train_original`
- How to prove that we can safely regard `id` as a useless column? [[Playground Series Season 3, Episode 3#^06bbfb|codes]] 
- If these two columns actually the similar things, then we can converge their names to `id`. [[Playground Series Season 3, Episode 3#^14eac2|codes]]



##### <mark style="background: #FFB8EBA6;">How cast dtypes of certain columns to save RAM</mark> 

- how to find out the max number of each numeric columns, [[Playground Series Season 3, Episode 3#^75aa71|codes]]
- what is the max number of all the numeric columns, [[Playground Series Season 3, Episode 3#^75aa71|codes]]
- cast all numeric columns to dtype Int16 and find the maximum value of each column, [[Playground Series Season 3, Episode 3#^75aa71|codes]]
- how to select only numeric columns to cast into Int16, and leave other columns unchanged, [[Playground Series Season 3, Episode 3#^75aa71|codes]]


##### <mark style="background: #FFB8EBA6;">If you have two datasets, how to find out shared and disjoined columns</mark> 

- How to use set.intersection and set.difference or set.symmetric_difference to find out shared and disjoint elements of columns of two dfs? [[Playground Series Season 3, Episode 3#^d2e019|codes]] 


##### <mark style="background: #FFB8EBA6;">How to make original and generated dataset look the same before joining</mark> 

- how to find out which dtypes are different when their column names are the same? [[Playground Series Season 3, Episode 3#^30a99d|codes]] 
- how to change dtype from binary String 'Yes' or 'No' to Int '1' or '0'? [[Playground Series Season 3, Episode 3#^51949f|codes]]
- how to make sure `original_samelooking` to have the columns in the same order as `train_generated`? [[Playground Series Season 3, Episode 3#^102a12|codes]] 
- prepare the feature column name list and target column name, [[Playground Series Season 3, Episode 3#^7f83e8|codes]] 


##### <mark style="background: #FFB8EBA6;">How to add a label to differentiate data source? How to join them</mark> 

- adding `is_generated` label to both generated and original datasets? [[Playground Series Season 3, Episode 3#^7fae97|codes]] 
- how to find out the max and min values of entire dfs and `int16` or alike? [[Playground Series Season 3, Episode 3#^a40e2f|codes]] 
- how to select columns by dtypes and cast into different dtypes before concating two dfs? [[Playground Series Season 3, Episode 3#^80d233|codes]]
- how to join the two dfs vertically? [[Playground Series Season 3, Episode 3#^a2390a|codes]] 


##### <mark style="background: #FFB8EBA6;">How to encode categorical columns</mark> 

polars [api](https://pola-rs.github.io/polars/py-polars/html/reference/series/api/polars.Series.cat.set_ordering.html)

- No need for a class like `MultiColumnLabelEncoder`, [[Playground Series Season 3, Episode 3#^40e4d1|codes]] 
- how to get the list of string columns for categorical features? [[Playground Series Season 3, Episode 3#^3442a8|codes]] 
- how to turn a categorical column into dummies columns? [[Playground Series Season 3, Episode 3#^be9fad|codes]] 🔥🔥🔥
- how to merge dummies columns into a single label encodered column? [[Playground Series Season 3, Episode 3#^be9fad|codes]] 🔥🔥🔥
- how to remove all dummies columns? [[Playground Series Season 3, Episode 3#^be9fad|codes]] 🔥🔥🔥


##### <mark style="background: #FFB8EBA6;">How to calc and remove all null or NAs: 0</mark> 

- How many null in each column [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]
- How many NAs or Nulls in each column  [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]
- How many NAs or Nulls in each row [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]
- How many null in total [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]


Read the codes and outputs in the twitter [thread](https://twitter.com/shendusuipian/status/1616440208492466183), Read everything on my repo [page](https://github.com/EmbraceLife/My_Journey_on_Kaggle/blob/main/Playground%20Series%20Season%203%2C%20Episode%203.md#first-things-to-check-about-the-dataset)

---




## <mark style="background: #FFB86CA6;">Challenges: smaller dataset and overfitting</mark> 

- Train: (1677, 35)
- Test: (1119, 34)
- examples/rows are too small compared to the number of features
- many models can easily overfit on such a small dataset
- What are we doing about it? #question  asked in [forum](https://www.kaggle.com/code/radek1/eda-training-a-1st-model-submission/comments#2110357) 




---

## <mark style="background: #FFB86CA6;">Training</mark> 


### <mark style="background: #FFB8EBA6;">What to import for using LightGMB </mark> 

#### <mark style="background: #ABF7F7A6;">For predicting employee attrition</mark>  [[Playground Series Season 3, Episode 3#^e3c58a|here]] 

- need `StratifiedKFold` to imbalance of positive vs negative labels
- using `LabelEncoder` to handle categorical columns, [api](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) 
- using `LGBMClassifier` for binary classification 
- using `roc_auc_score` metric

---


### <mark style="background: #FFB8EBA6;">How to do K-folds? </mark> 

#### <mark style="background: #ABF7F7A6;">How to StratifiedKFold</mark> 

- API of `StratifiedKFold` is different from `KFold`
- The key seems to be the target needs specified in `kf.split`.  [[Playground Series Season 3, Episode 3#^cb4b2f|codes]]
- How to split fold (10 folds for example) into (X_train, y_train) , (X_valid,  y_valid)   [[Playground Series Season 3, Episode 3#^cb4b2f|codes]]


- how to train and save models and scores [[Playground Series Season 3, Episode 3#^cb4b2f|codes]]
- how to display all feature importances? [[Playground Series Season 3, Episode 3#^0dc1f6|codes]]
- how to train with catboost with default? [[Playground Series Season 3, Episode 3#^5cf9d5|codes]]
- how to predict on test set with all models and ensemble the predictions? [[Playground Series Season 3, Episode 3#^56b6b7|codes]]
- how to make a submission? [[Playground Series Season 3, Episode 3#^42c6d2|codes]] 






---




#### <mark style="background: #FFB86CA6;">Evaluation Metrics</mark> 

Submissions are evaluated on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

From the wikipedia link, I found the following info to be helpful.

> The ROC curve is created by plotting the [true positive rate](https://en.wikipedia.org/wiki/True_positive_rate "True positive rate") (TPR) against the [false positive rate](https://en.wikipedia.org/wiki/False_positive_rate "False positive rate") (FPR) at various threshold settings.  

![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Roc_curve.svg/440px-Roc_curve.svg.png)

![{\displaystyle \mathrm {TPR} ={\frac {\mathrm {TP} }{\mathrm {P} }}={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FN} }}=1-\mathrm {FNR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/f02ea353bf60bfdd9557d2c98fe18c34cd8db835)

![{\displaystyle \mathrm {TNR} ={\frac {\mathrm {TN} }{\mathrm {N} }}={\frac {\mathrm {TN} }{\mathrm {TN} +\mathrm {FP} }}=1-\mathrm {FPR} }](https://wikimedia.org/api/rest_v1/media/math/render/svg/8f2c867f0641e498ec8a59de63697a3a45d66b07)

![{\displaystyle FPR={\frac {\mathrm {FP} }{\mathrm {FP} +\mathrm {TN} }}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/c5119dc2a74e72317ac2274c5b0d4d562597d8af)

---


read more on the repo page https://github.com/EmbraceLife/My_Journey_on_Kaggle/blob/main/Playground%20Series%20Season%203%2C%20Episode%203.md

---


#### <mark style="background: #FFB86CA6;">Important Notebooks to follow</mark> 

**Playground Series 3, Episode 1** 

###### Overview of two leading solutions from PS Season 3 Episode 1:

-   [1st place](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/377137) - AutoGluon with geospatial features
-   [2nd place](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/377179) - a blend of public solutions with personal ideas, including an NN trained in Keras!

What the NN design look like? 
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F11232701%2Fb00083ae121833a386890fe24a8e9ddb%2Fmodel%20(1).png?generation=1673332971801417&alt=media)

Resources mentioned to explore
- https://www.kaggle.com/code/dmitryuarov/ps-s3e1-coordinates-key-to-victory/
- https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html
- https://keras.io/guides/keras_tuner/getting_started/


---
---

## <mark style="background: #FFB86CA6;">Codes</mark> 


```python
# get the dataset 
train_generated = pl.read_csv('/kaggle/input/playground-series-s3e3/train.csv')
test_generated = pl.read_csv('/kaggle/input/playground-series-s3e3/test.csv')
train_original = pl.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
sample_sub = pl.read_csv('/kaggle/input/playground-series-s3e3/sample_submission.csv')
```

^146e83


```python
# How to prove that we can safely regard `id` as a useless column

(
    train_generated
    .select([
        pl.col('id'),
        pl.col('id').min().alias('min'),
        pl.col('id').first().alias('first'),        
        pl.col('id').max().alias('max'),                
        pl.col('id').last().alias('last'),
        (pl.col('id').diff().sum() == (train_generated.height-1)).alias('increase_by_1')
	    
    ])
)

(
    train_generated
    .select([
        pl.col('EmployeeNumber'),
        pl.col('EmployeeNumber').min().alias('min'),
        pl.col('EmployeeNumber').first().alias('first'),        
        pl.col('EmployeeNumber').max().alias('max'),                
        pl.col('EmployeeNumber').last().alias('last'),
        pl.col('EmployeeNumber').diff().alias('increase_by_1?'),  
    ])
)


```

^06bbfb

```python
# converge the two col names into the same, `id`
train_original = train_original.rename({'EmployeeNumber': 'id'})
```

^14eac2



```python
# how to find out which columns are Utf8/String or numeric or Int64
(
    pl.DataFrame({'columns': train_generated.columns, 
                  'dtypes': train_generated.dtypes, # How to find out the dtypes of each column? 
                  'utf8': [train_generated[name].is_utf8() for name in train_generated.columns],
                  'numeric': [train_generated[name].is_numeric() for name in train_generated.columns],                  
                  'Int64': [dtype == pl.Int64 for dtype in train_generated.dtypes],
                  'Int32': [dtype == pl.Int32 for dtype in train_generated.dtypes],                  
                 })
    .filter(pl.col('utf8') == True)
    # .filter(pl.col('Int64') == True)
    # .select('columns').to_series().to_list()
)

string_columns = (
    train
    .select([
        pl.col(pl.Utf8)
    ])
    .columns
)

numeric_columns = (
    train
    .select([
        pl.col(~pl.Utf8) # better than below
        # pl.col([pl.Int63, pl.Int32])
    ])
    .columns
)

```

^d02f8d


```python
# How many null in each column
train_generated.null_count()

# How many NAs or Nulls in each column
(
    train_generated
    .select([
        pl.all().is_null().sum()
    ])
)

# How many NAs or Nulls in each row
(
    train_generated
    .select([
        pl.all().is_null().cast(pl.UInt8)
    ])
    .sum(axis=1)
)

# How many null in total
train_generated.null_count().sum(axis=1)
```

^4a0fb0

```python
numeric_columns = (
    pl.DataFrame({'columns': train_generated.columns, 
                  'dtypes': train_generated.dtypes,
                  'utf8': [train_generated[name].is_utf8() for name in train_generated.columns],
                  'numeric': [train_generated[name].is_numeric() for name in train_generated.columns],                  
                  'Int64': [dtype == pl.Int64 for dtype in train_generated.dtypes],
                  'Int32': [dtype == pl.Int32 for dtype in train_generated.dtypes],                  
                 })
    .filter(pl.col('Int64') == True)
    .select('columns').to_series().to_list()
)

train_generated.describe() # to get all stats needed

(
    train_generated[numeric_columns]
    .max()
    .max(axis=1) # what is the max number of all the numeric columns
)


(
    train_generated[numeric_columns]
    .select([
        pl.all().max() # how to find out the max number of each numeric columns,
    ]) 
    .max(axis=1) # what is the max number of all the numeric columns
)

(
    train_generated[numeric_columns]
    .select([
        pl.all().cast(pl.Int16).max() # cast all numeric columns to dtype Int16 and find the maximum value of each column
    ]) 
    .max(axis=1) # what is the max number of the entire df
)


train_generated_optimized = (
    train_generated
    .with_columns([ 
        pl.col(numeric_columns).cast(pl.Int16) # how to select only numeric columns to cast into Int16, and leave other columns unchanged
    ]) 
)
train_generated.head()
```

^75aa71


```python
# how to get the list of string columns for categorical features
string_columns = (
    pl.DataFrame({'columns': train_generated.columns, 
                  'dtypes': train_generated.dtypes,
                  'utf8': [train_generated[name].is_utf8() for name in train_generated.columns],
                  'numeric': [train_generated[name].is_numeric() for name in train_generated.columns],                  
                  'Int64': [dtype == pl.Int64 for dtype in train_generated.dtypes],
                  'Int32': [dtype == pl.Int32 for dtype in train_generated.dtypes],                  
                 })
    .filter(pl.col('utf8') == True)
    .select('columns').to_series().to_list()
)
# the above is less optimal, below is the best
string_columns = (
    train
    .select([
        pl.col(pl.Utf8)
    ])
    .columns
)
```

^3442a8



```python
train_generated_dummies = (
    train_generated
    .to_dummies(columns=string_columns) # how to turn a categorical column into dummies columns
    .with_columns([ # merge dummies columns into a single label encodered column
        pl.concat_list('^'+string_column+ '.*$').arr.arg_max().alias(string_column) for string_column in string_columns
    ])
    .select([ # remove all dummies columns
        pl.all().exclude([('^' + string_column + '_.*$') for string_column in string_columns])
    ])
)
train_generated_dummies.head()
len(train_generated_dummies.columns)
```

^be9fad

```python
# How to use set.intersection and set.difference or set.symmetric_difference to find out shared and disjoint elements of columns of two dfs
inter = set(train_original.columns).intersection(set(train_generated.columns))
set(train_original.columns).symmetric_difference(set(train_generated.columns))
# {'EmployeeNumber', 'id'}
set(train_original.columns).difference(set(train_generated.columns))
# {'EmployeeNumber'}
set(train_generated.columns).difference(set(train_original.columns))
# {'id'}
set(train_original.columns).difference(inter)
# {'EmployeeNumber'}
set(train_generated.columns).difference(inter)
# {'id'}
```

^d2e019


```python
# how to find out which dtypes are different when their columns are the same?
(
    pl.DataFrame({
        'generated_cols': train_generated[list(inter)].columns,
        'generated_dtypes': [str(dt) for dt in train_generated[list(inter)].dtypes],
        'original_cols': train_original[list(inter)].columns,
        'original_dtypes': [str(dt) for dt in train_original[list(inter)].dtypes],
    })
    .filter(pl.col('generated_dtypes') != pl.col('original_dtypes'))
)
```

^30a99d

```python
# how to change dtype from binary String 'Yes' or 'No' to Int '1' or '0'
original_samelooking = (
    train_original
    .with_columns([
        pl.when(pl.col('Attrition') == 'No')
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias('Attrition')
    ])
)
```

^51949f

```python
# how to make sure original_samelooking to have the columns in the same order as train_generated
len(original_samelooking.columns) == len(train_generated.columns) == 35
original_samelooking[train_generated.columns].columns == train_generated.columns
original_samelooking = original_samelooking[train_generated.columns]
```

^102a12

```python
# prepare the features and target
features = train_generated.columns
features.remove('id')
features.remove('Attrition')
target = 'Attrition'
```

^7f83e8



```python
# source: https://stackoverflow.com/a/30267328/1105837

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
```

^40e4d1

```python
# import for LightGMB to handle categorical columns
# eliminating annoying lgbm warnings, source: https://stackoverflow.com/a/33616192/1105837
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
```

^e3c58a

```python
# how to train and save models and scores
clfs = []
scores = []
kf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

for i, (train_index, val_index) in enumerate(kf.split(train_encoded, y=train_encoded['Attrition'])): 
    X_train = train_encoded[features][train_index]
    X_val  = train_encoded[features][val_index]
    y_train = train_encoded[target][train_index]
    y_val = train_encoded[target][val_index]


#     clf = LGBMClassifier(n_estimators=150, categorical_feature=list(range(34)[-9:]), metric='auc') # actually worse result
    clf = LGBMClassifier(n_estimators=150, metric='auc')
    clf.fit(X_train.to_numpy(), 
            y_train.to_numpy(), 
            eval_set=[(X_val.to_numpy(), y_val.to_numpy())], 
            verbose=False)
    preds = clf.predict_proba(X_val.to_numpy())
    clfs.append(clf)
    scores.append(roc_auc_score(y_val, preds[:, 1]))
print(f'mean score across all folds: {np.mean(scores)}')
```

^cb4b2f

```python
# how to display all feature importances
for i in clf.feature_importances_.argsort()[::-1]:
    print(features[i], clf.feature_importances_[i]/clf.feature_importances_.sum())
```

^0dc1f6

```python
# how to train with catboost with default 
from catboost import CatBoostClassifier

scores = []
kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

for i, (train_index, val_index) in enumerate(kf.split(train_encoded, y=train_encoded['Attrition'].to_numpy())): # kf.split can work with pl.DataFrame
    X_train = train_encoded[features][train_index]
    X_val  = train_encoded[features][val_index]
    y_train = train_encoded[target][train_index]
    y_val = train_encoded[target][val_index]


    clf = CatBoostClassifier(iterations=200)
    clf.fit(X_train.to_numpy(),
            y_train.to_numpy(),
            eval_set=(X_val.to_numpy(),y_val.to_numpy()),
            verbose=False)
    
    preds = clf.predict_proba(X_val.to_numpy())
    clfs.append(clf)
    scores.append(roc_auc_score(y_val, preds[:, 1]))
print(f'mean score across all folds: {np.mean(scores)}')
```

^5cf9d5
```python
# how to predict on test set with all models and ensemble the predictions
test_preds = []

for clf in (clfs_f64pl + clfs_f64pl_cat):
    preds = clf.predict(test_pl_adddist[features].to_numpy())
	# preds = clf.predict_proba(test[features].values)
    test_preds.append(preds)

test_preds_mean_pl = (
    pl.DataFrame(test_preds)
    .transpose()
    .select([
        pl.all().explode()
    ])
    .mean(axis=1)
    .to_list()
)
```

^56b6b7

```python
# how to make a submission 
submission = pl.DataFrame({
    'id': test_pl.select('id').to_series(),
    'MedHouseVal': test_preds_mean_pl
})
# submission.head()

submission.write_csv('clfs_lgbm_cat_extsrc.csv')
```

^42c6d2


```python
# adding `is_generated` label to both generated and original datasets
original_samelooking = ( 
    original_samelooking
    .with_columns([
        pl.lit(0).alias('is_generated'), # add labels to distinguish two datasets
    ])
#     .select([
#         'id',
#         'is_generated'
#     ])
)

train_generated = ( 
    train_generated
    .with_columns([
        pl.lit(1).alias('is_generated'), 
    ])
)
```

^7fae97

```python
# how to find out the max and min of entire dfs and int16 or alike
np.iinfo(np.int16).min
np.iinfo(np.int16).max
np.iinfo(np.uint8).min
np.iinfo(np.uint8).max
train_generated.select([pl.col([pl.Int64, pl.Int32])]).max().max(axis=1)
original_samelooking.select([pl.col([pl.Int64, pl.Int32])]).max().max(axis=1)
```

^a40e2f

```python
# select columns by dtypes and cast into different dtypes before concating two dfs
train_generated = (
    train_generated
    .with_columns([
        pl.col([pl.Int64]).cast(pl.Int16),
        pl.col([pl.Int32]).cast(pl.UInt8),
        pl.col('Attrition').cast(pl.UInt8)
    ])
)

original_samelooking = (
    original_samelooking
    .with_columns([
        pl.col([pl.Int64]).cast(pl.Int16),
        pl.col([pl.Int32]).cast(pl.UInt8),
    ])
)
train_generated.head(2)
original_samelooking.head(2)
```

^80d233

```python
# how to join the two dfs vertically 
train = train_generated.vstack(original_samelooking) 
# train = pl.concat([train_generated, original_samelooking])
train.height == train_generated.height + original_samelooking.height
```

^a2390a
