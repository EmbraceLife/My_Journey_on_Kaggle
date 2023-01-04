<mark style="background: #FFB8EBA6;">How and Why shoud I get started</mark> 
- Introduced to me by Radek's [tweet](https://twitter.com/shendusuipian/status/1610269663262568448) 
- Radek's comp intro [video](https://www.youtube.com/watch?v=cIFRuaQy2Ow&loop=0) gets very interested in it
- Reasons not to do it: dataset is not real, therefore the problem is not serious
- Reasons to do it: time and energy will be focused on techniques and models, and learning will be more efficient (whereas in otto comp I have spent a month just to implement scripts before touching real complex models)

<mark style="background: #FFB8EBA6;">My plan for this comp</mark> 
- Implement everything Radek is sharing in this comp
- recording my [journey](https://forums.fast.ai/t/a-beginners-journey-to-playground-series-season-3-episode-1/103056) in fastai forum

<mark style="background: #FFB8EBA6;">📈 EDA + training a first model + submission 🚀</mark> 

<mark style="background: #FFB86CA6;">Data checking</mark> 
- more [info](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) on the dataset from sklearn
- what to predict? (*the the median house value for California districts*) [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115369488&cellId=1) 
- what are the independent variables? (8 or 9?) [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115369488&cellId=1)
- what is the evaluation metric? (root mean squared error and watch out for what) [cell](https://www.kaggle.com/code/radek1/eda-training-a-first-model-submission?scriptVersionId=115369488&cellId=1)
- is the column `id` a feature to be studied or just a artifact of preprocessingjust to be ignored? (use maximum occurrence of `id` to confirm) [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115387944&cellId=11)
- eyeball for numeric and categorical columns/features? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115387944&cellId=9)
- how many `NA`s or `null`s in each feature? [cell1](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115389565&cellId=15), [cell2](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115389565&cellId=16) 
- check the size of the `train` and `test`, [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115390387&cellId=18)

<mark style="background: #FFB86CA6;">Modeling</mark> 
- why do modeling, instead of doing statistical analysis to find interesting things? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115390387&cellId=19)
- 