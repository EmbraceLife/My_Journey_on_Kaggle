<mark style="background: #FFB8EBA6;">Life wisdom</mark> 
- [how to be happy](how to be [happy](https://muellerzr.github.io/blog/happiness.html) introduced to me by Radek's 1st newsletter in 2023)
- [exploit or explore](https://www.scotthyoung.com/blog/2019/09/17/explore-exploit/)


<mark style="background: #FFB8EBA6;">DL/ML</mark> 

<mark style="background: #FFB86CA6;">🚀🚀🚀 When should you give up on Machine Learning?</mark> [thread](https://twitter.com/radekosmulski/status/1611855146098782208)

- ML is good for you in all imaginable ways
- why I will Never give it up [reply](https://twitter.com/shendusuipian/status/1611871815856717824)

<mark style="background: #FFB86CA6;">🚀🚀🚀 Use <mark style="background: #FF5582A6;">Constraints</mark> To Learn Machine Learning 12x FASTER!</mark> [twitter thread](https://twitter.com/radekosmulski/status/1612016593789399045)  ^1a922c

- Figure out what is important 
	- <mark style="background: #ADCCFFA6;">training</mark>  and <mark style="background: #ADCCFFA6;">iterating</mark> models is the only most important thing
- Focus on it
- Forget the rest
	- <mark style="background: #BBFABBA6;">reduce</mark> time for anything else, e.g., hardware, courses, frameworks
	- dig into one framework until <mark style="background: #BBFABBA6;">proficiency in key concepts</mark> enable you to switch between frameworks with ease

<mark style="background: #FFB86CA6;">🚀🚀🚀 newsletter Jan 10th</mark> [link](https://mail.google.com/mail/u/0/#inbox/FMfcgzGrbvGQfkFLwqRMZzvxQGtVZHxJ)

> Focus on that one thing. Don't worry about what others do. You are the main protagonist in your own story!

> Don't listen to what worked for people who are vastly different to who you are.

> Be your own best thermostat. React to your current setting.

> Eliminate the bottleneck to make the whole system of you as a machine learning practitioner running much more smoothly!

- What's the one thing, the bottleneck for me?  😱😱😱
	- not build pipelines enough, not iterate pipelines enough
- How will I tackle this bottleneck? 🏇🏇🏇
	- on Kaggle, it's more like learning to build and iterate in the wild, I don't know what I will learn each day, if lucky I can find guides along the way, but in general it's more of keep exploring not knowing what is ahead
	- I learnt how to build and iterate fast from Kaggle comp like Playground Series, and I am super excited about it because I feel this is what I am missing.
	- on fastai part1, part2, wwf, I know what's ahead is systematic and promising to build me up as a proper practitioner, but the tasks are overwhemingly massive. 
	- My plan is to turn course notebooks and kaggle comps into building and iterating pipelines in which I will learn all the techniques of fastai in time.

<mark style="background: #FFB86CA6;">🚀🚀🚀 The Secret to Becoming a Data Scientist</mark> [video](https://youtu.be/yaEyxPdRkPI)

- Secret - Perseverence 🔥🔥🔥🔥🔥🔥🔥🔥🔥 ^893aa2
	- Perseverence is <mark style="background: #BBFABBA6;">not intensity</mark> which often goes against perseverence
	- Perseverence is <mark style="background: #BBFABBA6;">not discipline</mark> but to find your passion and emotions power at
	- Perseverence is the <mark style="background: #BBFABBA6;">ability to return to the same thing</mark> over weeks, months, years
		- first, pick the thing you are truly <mark style="background: #FF5582A6;">passionate</mark> about, which will sustain your perseverence
	- why perseverence is important
		- you need to give <mark style="background: #BBFABBA6;">time</mark> to learn and practice
		- it takes time to find the <mark style="background: #BBFABBA6;">right strategy</mark> to work for you

<mark style="background: #FFB86CA6;">🚀🚀🚀 Ask Me Anything by Radek</mark> tweet [thread](https://twitter.com/radekosmulski/status/1613031846606163968)

> Hi Radek, if you start over and just get started with Kaggle, and you heard an amazing Kaggler who won the first place in iMaterialist Challenge comp and willing to share everything on it. What would you do to make the most out of this comp and the Kaggler?

 🎥 <mark style="background: #FFB86CA6;">An intro to the Kaggle's OTTO RecSys competition</mark> [thread](https://twitter.com/radekosmulski/status/1613138626304708609)

- Just when I feel uncomfortable when not touching OTTO for a few days, Radek has come to rescue
- then I can ask question to AMA  as well



----
----

## <mark style="background: #FFB86CA6;">Radek's Youtube Channel</mark> 


- I have summarized Radek's video contents in terms of questions (answers provided)
- writing summaries like below takes longer than expected but helps me learn more than expected too 🎉
- Taking notes without implementing in codes ruins everything! I must make a notebook for every technique I learn from Radek! 🏗️

---

#### <mark style="background: #FFB8EBA6;">2 Things I Learned on Kaggle Today</mark>  [video](https://youtu.be/S7pv_aU_ER8)

^52405f

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

#### <mark style="background: #FFB8EBA6;">Feed your DL models up to 400x faster with the Merlin Dataloader! Fewer lines of code + better perf!</mark>  [video](https://youtu.be/Xyoa0r2QraI)

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

---

####  <mark style="background: #FFB8EBA6;">AMA with Radek Osmulski (learning ML, getting a DL job, Kaggling...)</mark>  [video](https://youtu.be/qjWZvdo7PYA)

^73faa3

[16:46](https://www.youtube.com/watch?v=qjWZvdo7PYA&t=1006s) How do you best learn from Kaggle and in particular experienced Kagglers that you meet there?
- keep doing what I am doing
[20:22](https://www.youtube.com/watch?v=qjWZvdo7PYA&t=1222s) How would you reconcile fast.ai and Merlin dataloaders?
- Merlin DataLoaders is for tabular dataset and problems
- Fastai dataloaders is for vision problems
[24:51](https://www.youtube.com/watch?v=qjWZvdo7PYA&t=1491s) How can I improve my skills in computer vision problems?
[32:30](https://www.youtube.com/watch?v=qjWZvdo7PYA&t=1950s) Overview of the ongoing RSNA competition and some tips and tricks.
[34:32](https://www.youtube.com/watch?v=qjWZvdo7PYA&t=2072s) What are some tips for creating a representative subset of a dataset? 🔥 
- check whether the scores of validation set of the models trained with subset are aligned with scores trained on full dataset ⚡💡
[41:43](https://www.youtube.com/watch?v=qjWZvdo7PYA&t=2503s) How to use a smaller dataset on Kaggle (or better, how to deal with huge dataset on Kaggle)
1. develop on a small subset for fast iteration ⚡💡
2. When we do developement with a subset, how fast should the model run in general? within a minute? ❓❓❓ <mark style="background: #BBFABBA6;">asked</mark> [here](https://twitter.com/shendusuipian/status/1613538430411247617)
3. when development is done, then move onto full dataset⚡💡
[36:39](https://www.youtube.com/watch?v=qjWZvdo7PYA&t=2199s) How can you find time for Kaggle competitions when you have a full-time job
- able to squeeze time more after a few more iterations
[39:30](https://www.youtube.com/watch?v=qjWZvdo7PYA&t=2370s) Can I become an ML engineer in 6 months?
[44:05](https://www.youtube.com/watch?v=qjWZvdo7PYA&t=2645s) Are you using GPT-3 to help write your newsletter?