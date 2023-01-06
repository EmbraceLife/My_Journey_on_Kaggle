
- I have summarized Radek's video contents in terms of questions (answers provided)
- writing summaries like below takes longer than expected but helps me learn more than expected too 🎉
- Taking notes without implementing in codes ruins everything! I must make a notebook for every technique I learn from Radek! 🏗️

---

<mark style="background: #FFB8EBA6;">2 Things I Learned on Kaggle Today</mark>  [video](https://youtu.be/S7pv_aU_ER8)
- 🚀 2 things for what? 
	- simple but effective techniques to improve your public scores on Kaggle comp
- 🎉 The 1st Thing, start [here](https://youtu.be/S7pv_aU_ER8?t=0)
	- what is the first technique?
		- adding <mark style="background: #FF5582A6;">labels for data sources</mark> by creating a new column
	- how does it work? 
		- if you are combining official dataset with external dataste for training
		- and if you add a column like `external` to distinguish official dataset from external dataset with labels `True` or `False`
		- your model will likely be trained with this new column added is likely to perform better
	- can it be applied to DL models? start [here](https://youtu.be/S7pv_aU_ER8?t=26)
		- yes, of course
	- 🤔 🌟 <mark style="background: #FF5582A6;">why</mark> this new column can help model to infer better? start [here](https://youtu.be/S7pv_aU_ER8?t=37)
		- this column can enable model to try to understand how the differences of two datasets affect performance
		- the model can give more appropriate predictions on the test set based on which data source it is from.
- 🎉 The 2nd Thing, start [here](https://youtu.be/S7pv_aU_ER8?t=59)
	- what's the 2nd technique?
		- adding a column (<mark style="background: #FF5582A6;">distance between house location and landmarks</mark> ) to your dataset
	- how does it work?
		- if you have longitude and latitude columns for house locations
		- if you have landmarks geoinfo which enables the calculation of the distance between your locations and the landmarks
		- the distance column can very likely improve your model's performance
	- 🤔 🌟 <mark style="background: #FF5582A6;">why</mark> ?
		- because domain knowledge tells us that important landmarks can influence the price of houses a lot
- resources
	- 🎯 Add data source [here](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376043)
	- 🗺️ Add distance from landmark feature [here](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376078)
	- 🧨 Awesome notebook on Geospatial Feature Engineering and Visualization [here](https://www.kaggle.com/code/camnugent/geospatial-feature-engineering-and-visualization)
	- 💥 Lat / Long Feature Engineering tricks from previous competitions [here](https://www.kaggle.com/competitions/playground-series-s3e1/discussion/376210)
- My implementations 😱😱😱
	- Notebook on 1st technique
	- Notebook on 2nd technique

---

<mark style="background: #FFB8EBA6;">Feed your DL models up to 400x faster with the Merlin Dataloader! Fewer lines of code + better perf!</mark>  [video](https://youtu.be/Xyoa0r2QraI)

- 🛠️ what does Merlin DataLoader library do and why Radek loves it? start [here](https://youtu.be/Xyoa0r2QraI?t=0)
	- feed DL models with datasets much fasters with much less code
- ⚡what other people say about Merlin DataLoader library? start [here](https://youtu.be/Xyoa0r2QraI?t=31)
	- 400 times faster than pytorch DataLoader 
	- An experienced user shared a kaggle [notebook](https://www.kaggle.com/code/cpmpml/matrix-factorization-with-gpu) (Matrix Factorization on GPU) demo on how much faster Merlin DataLoader is
	- more [notebooks](https://github.com/NVIDIA-Merlin/dataloader/tree/main/examples) to learn Merlin DataLoader
- 🦮 Radek walks us through a simpler notebook on how Merlin DataLoader work and stand out
	- what is `cudf` and why Radek likes it? start [here](https://youtu.be/Xyoa0r2QraI?t=119)
		- pandas on gpu with competent api, and works great with tabular dataset
	- what is the dataset used, start [here](https://youtu.be/Xyoa0r2QraI?t=159)
		- [otto](https://www.kaggle.com/competitions/otto-recommender-system) competition
	- how to prepare a simple dataset for Matrix factorization modeling, start [here](https://youtu.be/Xyoa0r2QraI?t=173)
		- how to create `next_aid` column alongside with `aid` and `next_session` column alongside with `session`?
			- use `shift(-1)`
		- how to clean the dataset to avoid `aid` from different sessions?
			- `filter(pl.col('session') == pl.col('next_session')` in polars
			- `[data.session == data.next_session]` in pandas
	- what can matrix factorization enable us to do? start [here](https://youtu.be/Xyoa0r2QraI?t=283)
		- create similarity scores
		- generate candidates
	- what does 99% ML scientists use `Dataset` and `DataLoader` to feed dataset to DL/ML models, start [here](https://youtu.be/Xyoa0r2QraI?t=305) (read screen code to answer question below)
		- write a class `ClicksDataset` to feed, access with idx and get length of a dataset
		- create two tensors `aid1` and `aid2`
	- how Radek test the speed of pytorch `Dataset` and `DataLoader`? start [here](https://youtu.be/Xyoa0r2QraI?t=381) (read screen code to answer question below)
		- how does Radek create pytorch dataset and dataloader with the simple datast we just built?
		- how many batches, and how many rows of each batch?
			- 2413 batches, 65536 rows for each batch
		- how to access the two tensors `aid1` and `aid2` of each batch?
		- how low is the GPU utilization?
	- why not writing faster code yourself? start [here](https://youtu.be/Xyoa0r2QraI?t=489)
		- not when there is a better, faster choice
- 😂 🚀 Radek walks with us on how to use Merlin DataLoader to feed the simple dataset, start [here](https://youtu.be/Xyoa0r2QraI?t=562)
	- does Merlin DataLoader require us to create `ClicksDataset` like in pytorch? No, nothing
	- how much faster is Merlin vs pytorch? 200 x faster
	- please see the code from the video [screen](https://youtu.be/Xyoa0r2QraI?t=644)
- What does Merlin DataLoaders mean to real-world, larger models, start [here](https://youtu.be/Xyoa0r2QraI?t=654)
	- better performance, optimization and shorter time
- 🎉 Why Radek is extremely excited about Merlin DataLoader, start [here](https://youtu.be/Xyoa0r2QraI?t=713)
	- it can scale up to more complex problems like data preprocessing using [NvTabular](https://github.com/NVIDIA-Merlin/NVTabular) library
- Why Merlin DataLoader is so much faster, start [here](https://youtu.be/Xyoa0r2QraI?t=876)
	- not use numpy, nor pytorch's Dataset, DataLoader
- More resources to learn Merlin DataLoader library, start [here](https://youtu.be/Xyoa0r2QraI?t=907)

