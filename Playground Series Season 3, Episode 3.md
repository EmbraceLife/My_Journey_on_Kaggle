# Tabular Classification with an Employee Attrition Dataset

Kaggle [site](https://www.kaggle.com/competitions/playground-series-s3e3) 


#### <mark style="background: #FFB86CA6;">Goal of Playground Series</mark> 

> the goals of the Playground Series remain the same—to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science. 

It means we should try as many models and techniques as we could and iterate as fast as we can

---


#### <mark style="background: #FFB86CA6;">What about generated dataset</mark> 

> These will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.

> Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. 

> While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have <mark style="background: #BBFABBA6;">far fewer artifacts</mark> . 

What are `artifacts` and why they sound like a bad thing?


---

#### <mark style="background: #FFB86CA6;">First things to check about the Dataset</mark> 


##### What does the official [dataset](https://www.kaggle.com/competitions/playground-series-s3e3/data?select=train.csv) look like

##### How many Feature columns: 33 
Excluding `id` and `Attrition`

> `Age`, `BusinessTravel`, `DailyRate`, `Department`, `DistanceFromHome`, `Education`, `EducationField`, `EmployeeCount`, `EnvironmentSatisfaction`, `Gender`, `HourlyRate`, `JobInvolvement`, `JobLevel`, `JobRole`, `JobSatisfaction`, `MaritalStatus`, `MonthlyIncome`, `MonthlyRate`, `NumCompaniesWorked`, `Over18`, `OverTime`, `PercentSalaryHike`, `PerformanceRating`, `RelationshipSatisfaction`, `StandardHours`, `StockOptionLevel`, `TotalWorkingYears`, `TrainingTimesLastYear`, `WorkLifeBalance`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`.

##### How many Categorical columns: 8
> `'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime'` 

How to find out the dtypes of each column? how to find out which columns are Utf8/String or numeric or Int64? [[Playground Series Season 3, Episode 3#^d02f8d|codes]] 

##### How many Numerical columns: 27 (including `id` and `Attrition`)
How to find out which columns are numeric or Int64? [[Playground Series Season 3, Episode 3#^d02f8d|codes]] 

##### How many Target column: 1, `Attrition`

##### How many column to ignore: 1,  `id`
- How to prove that we can safely regard `id` as a useless column? [[Playground Series Season 3, Episode 3#^06bbfb|codes]] 

##### How many null or NAs: 0
- How many null in each column [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]
- How many NAs or Nulls in each column  [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]
- How many NAs or Nulls in each row [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]
- How many null in total [[Playground Series Season 3, Episode 3#^4a0fb0|codes]]

##### How cast dtypes of certain columns to save RAM
- how to find out the max number of each numeric columns, [[Playground Series Season 3, Episode 3#^75aa71|codes]]
- what is the max number of all the numeric columns, [[Playground Series Season 3, Episode 3#^75aa71|codes]]
- cast all numeric columns to dtype Int16 and find the maximum value of each column, [[Playground Series Season 3, Episode 3#^75aa71|codes]]
- how to select only numeric columns to cast into Int16, and leave other columns unchanged, [[Playground Series Season 3, Episode 3#^75aa71|codes]]

Read the codes and outputs in the twitter [thread](https://twitter.com/shendusuipian/status/1616440208492466183)

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