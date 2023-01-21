# Tabular Classification with an Employee Attrition Dataset

Kaggle [site](https://www.kaggle.com/competitions/playground-series-s3e3) 


#### <mark style="background: #FFB86CA6;">My Goal of doing this comp</mark> 

> the goals of the Playground Series remain the same—to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. 

- I want to use this comp to build a <mark style="background: #D2B3FFA6;">template</mark> to help me to <mark style="background: #BBFABBA6;">try as many models and techniques as we could</mark> and <mark style="background: #BBFABBA6;">iterate as fast as we can</mark> 
- This <mark style="background: #D2B3FFA6;">template</mark> should show me<mark style="background: #BBFABBA6;"> all the steps</mark> to tackle a comp from start to finish, from surface scratch to fairly deep
- <mark style="background: #BBFABBA6;">Try as many models as I can</mark> using notebooks and discussions shared on Kaggle

---


#### <mark style="background: #FFB86CA6;">What about generated dataset</mark> 

> These will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

> Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. 

> While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have <mark style="background: #BBFABBA6;">far fewer artifacts</mark> . 

What are `artifacts` and why they sound like a bad thing?


---

#### <mark style="background: #FFB86CA6;">First things to check about the Dataset</mark> 

##### <mark style="background: #FFB8EBA6;">check out the dataset files</mark> [[Playground Series Season 3, Episode 3#^146e83|codes]] 

##### <mark style="background: #FFB8EBA6;">Overview of the official</mark> [dataset](https://www.kaggle.com/competitions/playground-series-s3e3/data?select=train.csv) 
- Train: (1677, 35)
- Test: (1119, 34)
- numeric columns: 25 (excluding `id` and `Attrition`)
- categorical columns: 8
- total feature columns: 33
- target column: 1
- columns to drop: 1, `id`

##### <mark style="background: #FFB8EBA6;">How many Feature columns</mark> : 33 
Excluding `id` and `Attrition`

> `Age`, `BusinessTravel`, `DailyRate`, `Department`, `DistanceFromHome`, `Education`, `EducationField`, `EmployeeCount`, `EnvironmentSatisfaction`, `Gender`, `HourlyRate`, `JobInvolvement`, `JobLevel`, `JobRole`, `JobSatisfaction`, `MaritalStatus`, `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `Over18`, `OverTime`, `PercentSalaryHike`, `PerformanceRating`, `RelationshipSatisfaction`, `StandardHours`, `StockOptionLevel`, `TotalWorkingYears`, `TrainingTimesLastYear`, `WorkLifeBalance`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`.

##### <mark style="background: #FFB8EBA6;">How many Categorical columns: 8</mark> 
> `'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime'` 

How to find out the dtypes of each column? how to find out which columns are Utf8/String or numeric or Int64? [[Playground Series Season 3, Episode 3#^d02f8d|codes]] 

##### <mark style="background: #FFB8EBA6;">How many Numerical columns: 27</mark> (including `id` and `Attrition`)
How to find out which columns are numeric or Int64? [[Playground Series Season 3, Episode 3#^d02f8d|codes]] 

##### <mark style="background: #FFB8EBA6;">How many Target column: 1</mark> , `Attrition`

##### <mark style="background: #FFB8EBA6;">How many column to ignore: 1</mark> ,  `id`
- How to prove that we can safely regard `id` as a useless column? [[Playground Series Season 3, Episode 3#^06bbfb|codes]] 

##### <mark style="background: #FFB8EBA6;">How many null or NAs: 0</mark> 
- How many null in each column [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]
- How many NAs or Nulls in each column  [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]
- How many NAs or Nulls in each row [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]
- How many null in total [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]

##### <mark style="background: #FFB8EBA6;">How cast dtypes of certain columns to save RAM</mark> 
- how to find out the max number of each numeric columns, [[Playground Series Season 3, Episode 3#^75aa71|codes]]
- what is the max number of all the numeric columns, [[Playground Series Season 3, Episode 3#^75aa71|codes]]
- cast all numeric columns to dtype Int16 and find the maximum value of each column, [[Playground Series Season 3, Episode 3#^75aa71|codes]]
- how to select only numeric columns to cast into Int16, and leave other columns unchanged, [[Playground Series Season 3, Episode 3#^75aa71|codes]]

##### <mark style="background: #FFB8EBA6;">Encode categorical columns</mark> 

polars [api](https://pola-rs.github.io/polars/py-polars/html/reference/series/api/polars.Series.cat.set_ordering.html)

- No need for a class like `MultiColumnLabelEncoder`, [[Playground Series Season 3, Episode 3#^40e4d1|codes]] 
- how to turn a categorical column into dummies columns? [[Playground Series Season 3, Episode 3#^be9fad|codes]] 🔥🔥🔥
- how to merge dummies columns into a single label encodered column? [[Playground Series Season 3, Episode 3#^be9fad|codes]] 🔥🔥🔥
- how to remove all dummies columns? [[Playground Series Season 3, Episode 3#^be9fad|codes]] 🔥🔥🔥


##### <mark style="background: #FFB8EBA6;">Shared and disjoined columns of two dfs</mark> 

- How to use set.intersection and set.difference or set.symmetric_difference to find out shared and disjoint elements of columns of two dfs? [[Playground Series Season 3, Episode 3#^d2e019|codes]] 

Read the codes and outputs in the twitter [thread](https://twitter.com/shendusuipian/status/1616440208492466183), Read everything on my repo page

---

#### <mark style="background: #FFB86CA6;">Challenges: smaller dataset and overfitting</mark> 

- Train: (1677, 35)
- Test: (1119, 34)
- examples/rows are too small compared to the number of features
- many models can easily overfit on such a small dataset




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


#### <mark style="background: #FFB86CA6;">What the submission format</mark> 

For each `EmployeeNumber` in the test set, you must predict the probability for the target variable `Attrition`. The file should contain a header and have the following format:

```
EmployeeNumber,Attrition
1677,0.78
1678,0.34
1679,0.55
etc.
```

read more on the repo page https://github.com/EmbraceLife/My_Journey_on_Kaggle/blob/main/Playground%20Series%20Season%203%2C%20Episode%203.md




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
train_generated = pl.read_csv('/kaggle/input/playground-series-s3e3/train.csv')
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


```

^06bbfb

```python
# How to find out the dtypes of each column? how to find out which columns are Utf8/String or numeric or Int64
(
    pl.DataFrame({'columns': train_generated.columns, 
                  'dtypes': train_generated.dtypes,
                  'utf8': [train_generated[name].is_utf8() for name in train_generated.columns],
                  'numeric': [train_generated[name].is_numeric() for name in train_generated.columns],                  
                  'Int64': [dtype == pl.Int64 for dtype in train_generated.dtypes],
                  'Int32': [dtype == pl.Int32 for dtype in train_generated.dtypes],                  
                 })
    .filter(pl.col('utf8') == True)
    # .filter(pl.col('Int64') == True)
    # .select('columns').to_series().to_list()
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