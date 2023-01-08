<mark style="background: #FFB8EBA6;">How and Why shoud I get started</mark> 
- Introduced to me by Radek's [tweet](https://twitter.com/shendusuipian/status/1610269663262568448) and this kaggle [post](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/375714)
- Radek's comp intro [video](https://www.youtube.com/watch?v=cIFRuaQy2Ow&loop=0) gets very interested in it
- 😱 😱 Reasons not to do it: 
	- dataset is not real, therefore the problem is not serious
- 😱 😂 🚀 ⭐ Reasons to do it: 
	- small and generated dataset which can be processed fast
	- iterate your pipelines fast
	- experimenting and learning new tricks fast
	- all large and complex datasets and comps can be first shrinked and then those benefits above can be applied

<mark style="background: #FFB8EBA6;">My plan for this comp</mark> 
- Implement everything Radek is sharing in this comp
- recording my journey in github [repo](https://github.com/EmbraceLife/My_Journey_on_Kaggle)

---
---

<mark style="background: #FFB8EBA6;">Interesting discussions</mark> 

- why 10 KF is not too many, see [discussion](https://www.kaggle.com/code/phongnguyen1/distance-to-cities-features-clustering/comments#2085057)


---
---


<mark style="background: #FFB86CA6;">Pipelines</mark> 

- EDA
	- numerical vs categorical
	- nulls, NAs
	- etc
- KF + training


---
---


<mark style="background: #FFB86CA6;">Milestone notebooks</mark>  and <mark style="background: #ADCCFFA6;">TODOS</mark> 
- train  LGBMRegressor model with official dataset alone:
	- [Radek](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115369488&cellId=19) (metric score: 0.56099, public score: 0.56237), 
	- [Daniel](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115556350) (float64 score: 0.56497, float32 score: 0.56506, public score: 0.56824)
- train and validate  LGBMRegressor model with combined dataset between official and additional:
	- [Radek](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115448636) (metric score: 0.52590225),
	- [Daniel](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115558134) (pandas score: 0.52590225, public score: 0.56097; polars float64: 0.525977, public score: 0.56064; polars float32: 0.525936, public: 0.56014) <mark style="background: #FF5582A6;">polars float32 outperform all</mark> 
- train LGBMRegressor with given parameters and KF 10 times from this [notebook](https://www.kaggle.com/code/soupmonster/simple-lightgbm-baseline) by @soupmonster:
	- [Radek](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115450828&cellId=19) (with random_state as 0, metric score: 0.519450), 
	- [Daniel](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115563958&cellId=24) (random_state as 19, f64 pandas: metric score: 0.52017, public score: 0.55846; f64 polars: metric: 0.52017, public: 0.55864; f32 polars: metric: 0.52003, public: 0.55858) <mark style="background: #FF5582A6;">polars f32 is worse than pandas f64 this time</mark> 
- train LGBMRegressor with specific tuning and KF 10 times + catboost regressor KF 10 times without tuning + two models have different random_state
	- [Radek](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115453015&cellId=24) : catboost mean metric score: 0.520077, public score: 0.55755, 
	- [Daniel](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115647583) : this round of submission shows <mark style="background: #FF5582A6;">float64 is better than float 32</mark> , and float 64 pandas and polars are the same. [version](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115626126) with submission files
		- clfs_f32pl_clfs_f32pl_cat: public: 0.55758, 
		- clfs_f64pd_cat
		- clfs_f64pd_clfs_f64pd_cat:  public: 0.55755 
		- clfs_f64pl_clfs_f64pl_cat:  public: 0.55755
- add `is_generated` column to the model above to distinguish external data source during training and inference [[Playground Series Season 3, Episode 1#^35e6ac|dive in]]
	- [Daniel](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115702196)
		- LGBMRegressor model metric score: 0.516229
		- Catboost metric score: 0.51463
		- ensemble public score: 0.55731 (increased 0.0002)
- to implement Radek's two things learnt today on Kaggle [tweet](https://twitter.com/radekosmulski/status/1610880953882406914) notebook <mark style="background: #ADCCFFA6;">todo</mark> 
- feature interactions notebook <mark style="background: #ADCCFFA6;">todo</mark> 
- hyperparameter search notebook <mark style="background: #ADCCFFA6;">todo</mark> 
- notebook to understand LGBMRegressor model <mark style="background: #ADCCFFA6;">todo</mark> 
- explore [catboost](https://catboost.ai/) library (maybe better than XGBoost and LightGBM and H2O) <mark style="background: #ADCCFFA6;">todos</mark> 

---
---

<mark style="background: #FFB8EBA6;">📈 EDA + training a first model + submission by @radek1 🚀</mark>  🔥 🧨 


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

---

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

--- 

<mark style="background: #FFB86CA6;">Modeling Radek version 7</mark>  [version 7](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115450828&cellId=19)

- The major changes of this version compared to version 6, see [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115450828&cellId=19) 
- 😱 😱 😱 but I don't know how did those parameters come from, nor how does LGBMRegressor model work
- how to set up all parameters in a dict for a function beforehand with `func(**params)`,  [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115563958&cellId=15), [cell2](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115563958&cellId=16)

<mark style="background: #FF5582A6;">Q&A</mark> 
- How did SoupMonster come up with the specified parameters for the model? asked and answered [here](https://www.kaggle.com/code/soupmonster/simple-lightgbm-baseline/comments#2087173), must give a try to Optuna <mark style="background: #BBFABBA6;">answered</mark> 

---

<mark style="background: #FFB86CA6;">Modeling Radek version 8</mark>  [version 8](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115453015)

- what's new in this version 8? (adding a new model `catboost` to mix), see [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115453015&cellId=23)
- how to build and train a catboost model, see [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115453015&cellId=24)
- what interesting came out of this new model, see [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115453015&cellId=25) 
- I have implemented catboost model built a few ensembles, [version](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission/notebook?scriptVersionId=115647583), and the submission files generated on this [version](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115626126) 


<mark style="background: #FF5582A6;">Q&A</mark> 

- what does `squared=False` in `mean_squared_error(y_val, preds, squared=False)` do? see answer [here](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115649311&cellId=11)
- 💥 ⭐ Why Radek consider catboost is doing amazingly even when its score is slightly worse than LGBMregressor? my hypothesis proposed and answered [here](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission/comments#2088071)
- why catboost can perform well equally as LGBMregressor with specific params after hyperparameter search? checkout [catboost.ai](https://catboost.ai/) 😍

---
---

<mark style="background: #FFB8EBA6;">Simple feature that boost your score +0.002 by @snnclsr</mark> 🔥


<mark style="background: #FFB86CA6;">Modeling Daniel version 37</mark>  [version 37](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115698631) ^35e6ac

- <mark style="background: #ADCCFFA6;">key idea</mark> : labeling the external datasource in a column
	- Radek shared the idea in a [video](https://youtu.be/S7pv_aU_ER8), detailed discussion about [this](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376043) 
- I implemented `is_generated` in this [version](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115702196) of notebook myself

---
---


<mark style="background: #FFB8EBA6;">S03E01: EDA for Modelling by @soupmonster</mark>  🔥 🔥 🔥 <mark style="background: #BBFABBA6;">not yet explored</mark> 

- <mark style="background: #ADCCFFA6;">key idea</mark>: Add distance from landmark feature, introduced to me by Radek in this [tweet](https://twitter.com/radekosmulski/status/1610880953882406914)
	- 🗺️ Add distance from landmark feature [here](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376078) explains why Longitude/Latitude are the most important Features
	- 🧨 Awesome notebook on Geospatial Feature Engineering and Visualization [here](https://www.kaggle.com/code/camnugent/geospatial-feature-engineering-and-visualization)
	- 💥 Lat / Long Feature Engineering tricks from previous competitions [here](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376210)


---
---


<mark style="background: #FFB8EBA6;">Distance to key locations by @phongnguyen1 version 26</mark> 🔥🔥🔥  [notebook](https://www.kaggle.com/code/phongnguyen1/distance-to-key-locations),his [dataset](https://www.kaggle.com/datasets/phongnguyen1/s03e01-california) including many geo info

- 😂 🧨 I reimplemented it in polars, see [version](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115729827)
	- what is the `haversine_distances` used to calculate distance between two locations, see [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115729827&cellId=15)
	- what are the 5 big cities used to calculate distances, see [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115729827&cellId=17)
	- how to create the distance columns, see [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115729827&cellId=18)
	- adding more distances columns to see whether it improves the results or not, see cell1, see result
- 😱😱😱 [More](https://www.kaggle.com/code/phongnguyen1/distance-to-key-locations?scriptVersionId=115730400&cellId=2) to explore in @phongnguyen1's notebook <mark style="background: #BBFABBA6;">todo</mark> 
	- reverse geo can give us more specified/useful locations to calc distance
	- also I would like to make more plotting to illustrate the locations

---
---


<mark style="background: #FF5582A6;">Todos</mark> 
- update with Radek's new versions
- watch Radek's new videos

---



