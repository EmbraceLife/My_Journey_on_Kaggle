<mark style="background: #FFB8EBA6;">How and Why shoud I get started</mark> 
- Introduced to me by Radek's [tweet](https://twitter.com/shendusuipian/status/1610269663262568448) 
- Radek's comp intro [video](https://www.youtube.com/watch?v=cIFRuaQy2Ow&loop=0) gets very interested in it
- Reasons not to do it: dataset is not real, therefore the problem is not serious
- Reasons to do it: time and energy will be focused on techniques and models, and learning will be more efficient (whereas in otto comp I have spent a month just to implement scripts before touching real complex models)

<mark style="background: #FFB8EBA6;">My plan for this comp</mark> 
- Implement everything Radek is sharing in this comp
- recording my [journey](https://forums.fast.ai/t/a-beginners-journey-to-playground-series-season-3-episode-1/103056) in fastai forum

<mark style="background: #FFB8EBA6;">📈 EDA + training a first model + submission 🚀</mark> 

<mark style="background: #FFB86CA6;">Milestone notebooks</mark> 
- train  LGBMRegressor model with official dataset alone: [Radek](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115369488&cellId=19) (metric score: 0.56099, public score: 0.56237), [Daniel](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115556350) (float64 score: 0.56497, float32 score: 0.56506, public score: 0.56824)
- train and validate  LGBMRegressor model with combined dataset between official and additional: [Radek](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115448636) (metric score: 0.52590225), [Daniel](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115558134) (pandas score: 0.52590225, public score: 0.56097; polars float64: 0.525977, public score: 0.56064; polars float32: 0.525936, public: 0.56014) <mark style="background: #FF5582A6;">polars float32 outperform all</mark> 
- train LGBMRegressor with given parameters from this [notebook](https://www.kaggle.com/code/soupmonster/simple-lightgbm-baseline) : [Radek](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115450828&cellId=19) (with random_state as 0, metric score: 0.519450),  [Daniel](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115563958&cellId=24) (random_state as 19, f64 pandas: metric score: 0.52017, public score; f64 polars: metric: 0.52017, public; f32 polars: metric: 0.52003, public: 0.55858)
- feature interactions notebook
- hyperparameter search notebook
- notebook to understand LGBMRegressor model


<mark style="background: #FFB86CA6;">Data checking version 5</mark>   [version 5](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115370967)

- more [info](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) on the dataset from sklearn
- what to predict? (*the the median house value for California districts*) [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115369488&cellId=1) 
- what are the independent variables? (8 or 9?) [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115369488&cellId=1)
- what is the evaluation metric? (root mean squared error and watch out for what) [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115369488&cellId=1)
- is the column `id` a feature to be studied or just a artifact of preprocessingjust to be ignored? (use maximum occurrence of `id` to confirm) [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115387944&cellId=11)
- eyeball for numeric and categorical columns/features? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115387944&cellId=9)
- how many `NA`s or `null`s in each feature? [cell1](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115389565&cellId=15), [cell2](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115389565&cellId=16) 
- check the shape of the `train` and `test`, [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115390387&cellId=18)

<mark style="background: #FFB86CA6;">Modeling version 5</mark> 
- why do modeling, instead of doing statistical analysis to find interesting things? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115390387&cellId=19)
- what are all the libraries and funcions needed for the modeling? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115426784&cellId=21)
- how to learn more of the classes and functions imported for modeling? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115426784&cellId=21)
- what are the features and target? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115426784&cellId=24)
- why adding additional dataset and what is the additional dataset? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115426784&cellId=25)
- how to download the additional dataset? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115426784&cellId=26)
- check the dataset (as a dict) provided by `fetch_california_housing` as a dict? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115427577&cellId=28)
- how to concat the features (numpy.array) and target (numpy.array) from the dict? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115427577&cellId=30)
- how to split the `train` into 5 folds and control the randomness of each fold with `KFold(n_splits=5, random_state=0, shuffle=True)`? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115430766&cellId=31)
- what exactly can `for i, (train_index, val_index) in enumerate(kf.split(train)):` give us? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115432290&cellId=34)
- how to access each fold of `X_train`, `X_val`, `y_train`, `y_val` with a list of features and a list of idx? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115432650&cellId=36)
- reading docs of `LGBMRegressor`, [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115433459&cellId=29)
- how to create a `LGBMRegressor` model with specific `learing_rate` and `n_estimators`, `metric`? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115433459&cellId=31)
- how to add additional dataset to the model for training? [pandas-numpy](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115441212&cellId=32) vs [polars](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115441212&cellId=34)
- make sure all the data inputs are the same shape between pandas/numpy version and polars version, [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115441212&cellId=30)
- when `LGBMRegressor` does `fit`, the data inputs should mostly be numpy array (sometimes pandas, but not polars), see Radek's pandas [version](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115441212&cellId=32), my polars2numpy [version](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115441212&cellId=34)
- reading docs of `LGBMRegressor.fit`, [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115441212&cellId=36)
- how to make predictions on a `X_val` with `model.predict`? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115449369&cellId=32)
- how to calculate metric score with `mean_squared_error(truth, pred, squared=False)`? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115449369&cellId=32)
- how to take the mean of a list with `pl.Series(list).mean`? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115449369&cellId=32)
- how to rank the importance of different features/columns from the trained model with `clf.feature_importances_`? with [pandas](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115449369&cellId=31) vs [polars](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115449369&cellId=35)
- what insight can the feature importance offer us? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115449369&cellId=36)
- what is the shortcomings of tree based models? (feature interactions, and what to do about it) [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115449369&cellId=36) 
- how to make 5 predictions from 5 models and put them into a list? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115455989&cellId=37)
- how to take the mean from the 5 predictions (ensemble) with `transpose`, `explode`, `mean(axis=1)`? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115455989&cellId=43)
- how to build a dataframe from `id` and ensembled prediction with `pl.DataFrame` and save it to csv `write_csv`? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115455989&cellId=45)

<mark style="background: #FF5582A6;">Q&A version 5</mark> 
- How `learning_rate` and `n_estimators` of `LGBMRegressor` get chosen? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115433459&cellId=30), asked [here](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission/comments#2085783) <mark style="background: #BBFABBA6;">direction provided</mark> 
- Question: pandas version or polars version trained twice, the scores are very close but not the same, why?  [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115440584&cellId=31) asked and answered [here](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission/comments#2085777) figured out [here](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115474793&cellId=32) <mark style="background: #BBFABBA6;">solved</mark> 
- why scores produced by pandas and polars are different? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115440584&cellId=31) asked and answered [here](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission/comments#2085777) <mark style="background: #BBFABBA6;">explored but solved</mark> 
- Will Radek dig into the interactions between features? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115449369&cellId=36) asked and answered [here](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission/comments#2085794) <mark style="background: #BBFABBA6;">direction provided</mark> 


<mark style="background: #FFB86CA6;">Modeling version 6</mark>  version [6](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115450828) : 

- changes of Radek's in version 6
	- merge the additional dataset with official dataset [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115448636&cellId=18)
	- changed the `random_state` to 19 from 0,  
	- metric score changed to 0.5259 from 0.5609. (I suspect it is due to the change of dataset) [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115448636&cellId=19)
- implement the above in polars in my notebook
	- pandas: join additional dataset with competition dataset (see [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115521879&cellId=15)) and feed 5 fold split in the numpy array with `to_numpy` to the training, see [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115521879&cellId=16)
	- polars: join additional dataset with competition dataset (see [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115521879&cellId=22)) and feed 5 fold split in the numpy array with `to_numpy` to the training, see [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115521879&cellId=26)
	- I compared and proved all data inputs for pandas and polars are the same as df or series (see [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115532182&cellId=33)), all arrays are the same too when same dtype enforced (see [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115532182&cellId=35))
	- 😂 🚀 🎉 I figured out why pandas and polars training have different results under same randomness: the same <mark style="background: #FF5582A6;">dtypes</mark> must be <mark style="background: #FF5582A6;">enforced</mark> , see [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115532182&cellId=36)
		- 😱 ⭐ but only the same when enforced into `pl.Float32`, slightly difference when `pl.Float64`, see [notebook](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115551847) 
		- 😂 🎉 using official dataset alone both `float64` and `float32` are all the same, see [notebook](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115556350)

<mark style="background: #FF5582A6;">Q&A Radek version 6 vs my version 22</mark> 
- why scores produced by pandas and polars are different? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115440584&cellId=31) asked and answered [here](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission/comments#2085777) and solved in this notebook [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115532182&cellId=36) <mark style="background: #BBFABBA6;">solved</mark>
- why is the falling scores from Radek's version 5 to version 6? my guess is that after joining the additional dataset, the problem gets harder due to more data. [asked](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission/comments#2086984) here <mark style="background: #BBFABBA6;">hypothesis proposed</mark> 


<mark style="background: #FFB86CA6;">Modeling Radek version 7</mark>  [version 7](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115450828&cellId=19)

- The major changes of this version compared to version 6, see [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115450828&cellId=19) 
- 😱 😱 😱 but I don't know how did those parameters come from, nor how does LGBMRegressor model work
- how to set up all parameters in a dict for a function beforehand with `func(**params)`,  [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115563958&cellId=15), [cell2](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115563958&cellId=16)

<mark style="background: #FF5582A6;">Q&A</mark> 
- How did SoupMonster come up with the specified parameters for the model? asked and answered [here](https://www.kaggle.com/code/soupmonster/simple-lightgbm-baseline/comments#2087173), must give a try to Optuna <mark style="background: #BBFABBA6;">answered</mark> 

---
<mark style="background: #FF5582A6;">Todos</mark> 
- What if I enforce dataset's dtype as float64 or int 32 to see the differences in scores? 
- what if the additional dataset is not added? why adding it is more interesting?
- update with Radek's new versions
- watch Radek's new videos

---



