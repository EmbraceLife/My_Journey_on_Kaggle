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

## <mark style="background: #FFB8EBA6;">2 Things I Learned on Kaggle Today</mark>  [video](https://youtu.be/S7pv_aU_ER8)

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

## <mark style="background: #FFB8EBA6;">Feed your DL models up to 400x faster with the Merlin Dataloader! Fewer lines of code + better perf!</mark>  [video](https://youtu.be/Xyoa0r2QraI)

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

##  <mark style="background: #FFB8EBA6;">AMA with Radek Osmulski (learning ML, getting a DL job, Kaggling...)</mark>  [video](https://youtu.be/qjWZvdo7PYA)

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


---
## <mark style="background: #FFB8EBA6;">Radek introduce OTTO comp</mark>  [Thread](https://twitter.com/radekosmulski/status/1613138626304708609)

^c32318

In this video, we meet several very powerful RecSys concepts that can be extended to real-life scenarios! 
- [0:00](https://www.youtube.com/watch?v=gtPEX_eRAVo&t=0s) Competition intro 
- [1:09](https://www.youtube.com/watch?v=gtPEX_eRAVo&t=69s) Data overview 
- [6:40](https://www.youtube.com/watch?v=gtPEX_eRAVo&t=400s) Baseline: last 20 aids as prediction
- [8:54](https://www.youtube.com/watch?v=gtPEX_eRAVo&t=534s) Co-visitation matrix: how to improve upon the baseline?
- [12:53](https://www.youtube.com/watch?v=gtPEX_eRAVo&t=773s) Candidate reranking using static rules
- [20:25](https://www.youtube.com/watch?v=gtPEX_eRAVo&t=1225s) Second-stage Ranker 
- [22:15](https://www.youtube.com/watch?v=gtPEX_eRAVo&t=1335s) Word2vec 
- [28:39](https://www.youtube.com/watch?v=gtPEX_eRAVo&t=1719s) Matrix Factorization
- [32:05](https://www.youtube.com/watch?v=gtPEX_eRAVo&t=1925s) What's next The presented concepts build on one another. 
- Furthermore, the OTTO dataset is a very interesting and elegant one, as many features can be created from essentially 4 columns! 

- Competition overview: [https://www.kaggle.com/competitions/o...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa0J6U1FIZ3ZhWk0wWEJucUtLRFhydV9KNDZ6d3xBQ3Jtc0ttZmxZZ1VZY2ZWN2xEWTFXdmEzY2gyNHpFTTVwUGVhdHk2SmtUbGFNRFNBb0VnZFZwcWRPaXdfMC1WRHJoYmdKNTFOSFRUQUtBZjBfWnZoN0FuUWRDQ3Uzc0dTSHpvY3NVYURmQWk4cmFFMUVWcDNPbw&q=https%3A%2F%2Fwww.kaggle.com%2Fcompetitions%2Fotto-recommender-system&v=gtPEX_eRAVo) 
- And here are the notebooks mentioned in the video: 
- 📈 [EDA] A look at the data + training splits: [https://www.kaggle.com/code/radek1/ed...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbno4YzM4bUxJb0EtdGFqbWdFSENYeWgxWjdEZ3xBQ3Jtc0ttdGZ5YUxYNkRuNktsVmVFRFlMTmZpNEVpZXNHSGVXLW5kakUzMkV2TXJaWE9hWEZwVDVYVU1LTUtrMmFxYjl3NEhRTmVSQU43QW42ZmtfMEdjbkpIZGhCdUJITDAwaHdYWkstQXBGS1VXRnJqRU9YWQ&q=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fradek1%2Feda-a-look-at-the-data-training-splits&v=gtPEX_eRAVo) 
- Baseline ([https://www.kaggle.com/code/radek1/la...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGp2dENwc1loZWZiaWpiY2ppakJuYUhWbXhVZ3xBQ3Jtc0tsQ2Z3bnd3NExVR2t2a0dlTThGZVA0dnNNaWRNVHFwc09qOVVKOFpabzhPdEFvUTBHNW9fTVlBOEVsYlZIRlB3MGlSSVlubDlMbV9nQUdiMjhuakp4NDNkWEFyX0xyYVVwMTVnTUxkM01IbkppSDA0RQ&q=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fradek1%2Flast-20-aids%29%3A&v=gtPEX_eRAVo) [https://www.kaggle.com/code/radek1/la...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbm5pMTR1VHl1UnhKVUlreFNLLV9BRGNxS2lad3xBQ3Jtc0tsSDRMYjhLMzdGN1pvcFNXWjl6RG9NVDg4TWRJMS02TlhoUG1Jc1JEQWk4U0ZrUkdRZE5IYUJCMFk1ZTNhZk1PenZUaEp1MVBBdGdBa2R3TDBBa3EtbVNUXzNDdXhtdnFXXzBNa3pqS1kyalNKNTlNZw&q=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fradek1%2Flast-20-aids&v=gtPEX_eRAVo) 
- Co-visitation Matrix: [https://www.kaggle.com/code/vslaykovs...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbWRkQV9aa1lrSUp4Y0xPNzRPdWE4Nmd4SU41QXxBQ3Jtc0traFpLbms1SDBkWTNiQ1FuWFN0dC1ZSlRRQVZFRjI2UzdEMnM5SEU5dnBkTHNoSG5fdWVtMWl5Smc5a1daVHpYSEZEM0g2Z0p0amFfM29XWkZkS0VCUkFzMkxVeldMZkxHcV9zMFBlU2RIMVNZUEN5aw&q=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fvslaykovsky%2Fco-visitation-matrix&v=gtPEX_eRAVo) 
- Candidate ReRank Model - [LB 0.575]: [https://www.kaggle.com/code/cdeotte/c...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbmFMbklITlNDYktBUURVVlhINEQ4QlFrbFNmUXxBQ3Jtc0ttcW03cjJqclZnTkxqSzFIb1hQQ2NBTGxycDI0RHhxMURmTE5meXNUVjZSZHNWY1RZb3ZObWhDMGZoWE5aU0ZuUk1uTUI1X1FWUFo1Zno4aWZGMzd5emRacmNjYy1VT0lONDdNNE4tYjhZLTB0OEw3cw&q=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fcdeotte%2Fcandidate-rerank-model-lb-0-575&v=gtPEX_eRAVo) 
- 💡 [polars] Proof of concept: LGBM Ranker🧪🧪🧪: [https://www.kaggle.com/code/radek1/po...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa3RCNEIxUEZua0lYcFZWamExNkF0S2FxMFRJd3xBQ3Jtc0tucTRwaElnN212bUd0N0ljQ1RtbVhhUzE1NUpkOHlVWDNLS0RtRnFfQ1JGWjVLYnUtX083VEh0SHJHMUVTVHF2WWlTQmhPZDhkai1iR29TUHpBY1JSNlJnVVp2TFNMa3N4MzZJeWFqU3J0T25GS0Fadw&q=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fradek1%2Fpolars-proof-of-concept-lgbm-ranker&v=gtPEX_eRAVo) 
- 💡 Word2Vec How-to [training and submission]🚀🚀🚀: [https://www.kaggle.com/code/radek1/wo...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbUFuT0Z2OWUzRWpEdzE5a1ZWSnFnVUNkRXA4Z3xBQ3Jtc0tuOFUzUEFaZDRZQXRYNWtDVmR5TDlnNGs4bTNDaDdFZEhfSnEycjQwT0V0cFBLeEVYTzBaUVVzMjhiUmh6TXZwRUZsMkk1Vk1DbEFVRTZkVjJFcjJBNWFtRXNPSF9OTjItV1l3a3ZwMmpGRTYzd244cw&q=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fradek1%2Fword2vec-how-to-training-and-submission&v=gtPEX_eRAVo) 
- 💡Matrix Factorization [PyTorch+Merlin Dataloader]: [https://www.kaggle.com/code/radek1/ma...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqblFtQ21SSW40RjBZVGM0Y1EyRC1NTFBvdFd0UXxBQ3Jtc0tuRTVCbnVPVFduVjNZVjktZHFzZW1yNzB2dzNSb0g3TS1uSG13ZUNPcHd5LXdGVFVwZTF1VWprdkEySkhURWt6THBXdjN5dWRmUkRSckVmNHp5UGNEVHVVUDlUcl9Ncm9nRWpnc3oxQkNoSTVWd2JURQ&q=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fradek1%2Fmatrix-factorization-pytorch-merlin-dataloader&v=gtPEX_eRAVo) 

- 🎥 Subscribe to my YouTube channel to learn more: [https://www.youtube.com/channel/UCgdd...](https://www.youtube.com/channel/UCgddGFFOLmhHSq_zhoa5SUQ) 
- 📗 My book on learning Machine Learning: [https://rosmulski.gumroad.com/l/learn...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbl9JanlJOFB4OXBmb3ZnQUo1RXhubG5YMEU1QXxBQ3Jtc0ttX0tVSUNHRWFRNmtUOXNlcGYyMklsWllhd1FGaUlGUW94Rkxfb2Y3dFFSVGNxWEd0Q0o3QnZ3NDlaQ0JrVUNCV3FrSXZBUGo4cERHUHJ5NTMxcDdzQTBoRVluaGFtV0V6amJtb2EzOGRFdFhpaFY4bw&q=https%3A%2F%2Frosmulski.gumroad.com%2Fl%2Flearn_machine_learning%2Fyoutube&v=gtPEX_eRAVo) 
- 📬 My newsletter: [https://newsletter.radekosmulski.com/](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa3BiZzhKMVpfQk92UW9ZcWVyd1RxTExxX2JiUXxBQ3Jtc0trYnludmo5S3UyYno5ZW9mN3k1VC1Zb2dyQXViYnV4eWRsdDFyVGg0cUdRYVdrNEJMQUtub1pGVUgtRzBHT3hGNVR6WmZLenhHY3MtNjZpaTIwRlN3dHpRSjRGY2xDVDU4c0d6SE9NMmNrcHpPVnVEMA&q=https%3A%2F%2Fnewsletter.radekosmulski.com%2F&v=gtPEX_eRAVo) 
- 📒 My blog: [https://radekosmulski.com/](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa2NoaGl0M2hKOV9HWldpbUxJdVNpSDc1ZGdxZ3xBQ3Jtc0tudG54dzlaNGc0UXFja1VUOTFyN29SZk1DN2pFRllCNEo1MVNQWE5YRFFVcVBZN0dNS0JXMURfVkswZGY2M0kwUXlTMFUxMHMtM2poenhSaTEybUpVLXRSWDRIWXJjQ2JLTHhSdHN4aW1qaEN0RWkzYw&q=https%3A%2F%2Fradekosmulski.com%2F&v=gtPEX_eRAVo)


---

## <mark style="background: #FFB8EBA6;">Radek talking about MasterMind group</mark>   [Thread](https://twitter.com/radekosmulski/status/1615286309974867974)

- what is mastermind group and why it can do magic to you? [video](https://youtu.be/KEVeMunS0Jc?t=29) ^c04b53
	- learn the know-how from the group
	- emotional drive to sustain 
	- reflection of your learning can benefit yourself and contribute to your mastermind group 
- how to form your mastermind group? [video](https://youtu.be/KEVeMunS0Jc?t=285)
	- same level of experience as you
	- you can teach each other and benefit each other with different perspectives
	- find them online through interactions on fastai and kaggle

---

## <mark style="background: #FFB8EBA6;">Playground Series S3E03 Competition on Kaggle | Intro</mark> 

- what's interesting about the dataset? [video](https://youtu.be/8CO7FnF2yNM?t=0)
- what about variables columns, what's in a row, and what to predict ? [video](https://youtu.be/8CO7FnF2yNM?t=73) 
- what is the metric? area of ROC curve, why it is a good metric for this comp? [video](https://youtu.be/8CO7FnF2yNM?t=100)
- How all boosted tree models handle string/categorical columns? and what extra benefits does lightGBM provide us? [video](https://youtu.be/8CO7FnF2yNM?t=125)
- What kind of challenge does this datast bring to us, train set (1677 rows, 35 columns), test set (1117, 34)? overfitting  [video](https://youtu.be/8CO7FnF2yNM?t=202)
- Any missing values? - [video](https://youtu.be/8CO7FnF2yNM?t=230)
- Adding original data which generated the comp official dataset and adding a column to distinguish original from official dataset gives model great improvement - [video](https://youtu.be/8CO7FnF2yNM?t=243) 
- setup for feature columns and target column - [video](https://youtu.be/8CO7FnF2yNM?t=373)
- Benefit and Warning of using `MultiColumnLabelEncoder` to convert string/categorical columns into integer columns- [video](https://youtu.be/8CO7FnF2yNM?t=406)
- Transform train and test sets with `MultiColumnLabelEncoder`- [video](https://youtu.be/8CO7FnF2yNM?t=491)
- Use `StratifiedKFold` to spread out positive labels equally into 10 folds for training and validation in order to handling the unbalanced positive and label labels (5%vs95%) - [video](https://youtu.be/8CO7FnF2yNM?t=528)
- How Radek use primitive hyperparameter search method to get to `n_estimator = 150` when creating a LightGBM model and what about advanced hyperparameter search? - [video](https://youtu.be/8CO7FnF2yNM?t=594)
- How to create a LightGBM model by specifying categorical features, train the model, and use model to predict probabilities for negative and positive outcomes, and how to calc the ROC ? - [video](https://youtu.be/8CO7FnF2yNM?t=636)  
- How to find out the most important features and their importances through the trained model instead of doing EDA? What the feature importances could not tell us? - [video](https://youtu.be/8CO7FnF2yNM?t=725)
- Why add Catboost models and how they perform even better? - [video](https://youtu.be/8CO7FnF2yNM?t=871)
- How to do ensemble of all the models above and make a submission? - [video](https://youtu.be/8CO7FnF2yNM?t=941)

#### But what to do next? 
- There are some todos which I planned in the first playground series introduced by Radek
- We should definitely read other kaggler's notebooks and discussions for new ideas to try
- but what does Radek recommend? 

---

## <mark style="background: #FFB8EBA6;">Radek Newsletter: Is AI overhyped?</mark>  2023.1.17

> Many activities are worth doing, worth participating in, <mark style="background: #BBFABBA6;">even if you never stand a chance to be the best in the world at them</mark> 

> So is AI overhyped? Yes, as evidenced by the silly tweets, there is a large portion of hype that will always be there and that will always be misguided regardless of how good the underlying technology is.

---

## Radek YT channel: <mark style="background: #FFB8EBA6;">Parallels between learning Machine Learning and starting a YouTube Channel</mark> 

- get started [video](https://youtu.be/4HM8MCIzu-g?t=56) 
- Thanks Radek, Now I know It's ok that
	- I don't have to believe I could make a living out of doing DL/ML, but just focusing on pursuing my passion and interest in ML/DL. 
	- I sometimes get unmotivated and losing steam and do less work, but just keep protecting and curating my interest and keep doing work
- What's Radek's advices if my dream is to apply ML/DL to my life and the real world?
	- train, evaluate and deploy your model and keep iterating with kaggle competitions and github repos
	- don't worry about phDs and learning and developing theories, the best talents who are doing cutting edge work are not necessarily the phDs.
	- just get hands dirty and keep tinkering and push the boundary forward


---
## Radek YT channel: <mark style="background: #FFB8EBA6;">How to run Merlin RecSys tutorials on Colab for FREE!</mark> 

Tutorial [video](https://youtu.be/YHFJj8-ik-8), and Tutorial notebooks [repo](https://github.com/NVIDIA-Merlin/models/tree/main/examples) 
- Reading the comments of the notebooks to learn more of the library
- get start with the first notebook on [colab](https://colab.research.google.com/drive/1zJPZrraNcPbo8DYrAWj4dmHxdWdlGs1p?usp=sharing) 
- get GPU runtime [video](https://youtu.be/YHFJj8-ik-8?t=103)
- how to install merlin on colab [video](https://youtu.be/YHFJj8-ik-8?t=209) and the [code](https://medium.com/nvidia-merlin/how-to-run-merlin-on-google-colab-83b5805c63e0) needed 🔥🔥🔥
- what libraries can't be run on colab [video](https://youtu.be/YHFJj8-ik-8?t=324) 
	- a lib efficiently serving DL models, created by Nvidia, but need your own server, so not run on colab
- how does merlin come in? how does mergin do with serving? [video](https://youtu.be/YHFJj8-ik-8?t=372)
- import and download dataset and an intro to movielens dataset, what is merlin dataset, [video](https://youtu.be/YHFJj8-ik-8?t=442)
- training, evaluating the model and checking the metrics, [video](https://youtu.be/YHFJj8-ik-8?t=539) 
- a great exercise: to try different values of hyperparameters of `DLRMModel`, [video](https://youtu.be/YHFJj8-ik-8?t=756)
	- I tried 3 different `embedding_dim`
- move on to the next notebook, colab [notebook](https://colab.research.google.com/github/NVIDIA-Merlin/models/blob/main/examples/02-Merlin-Models-and-NVTabular-integration.ipynb#scrollTo=I3xKZO8hRpJi), but have some issues with the proper dataset to use,  [video](https://youtu.be/YHFJj8-ik-8?t=845)
- get started on the second notebook (problem solved) [video](https://youtu.be/YHFJj8-ik-8?t=1156)
	- will show us how NvTabular and Merlin work together
- Wow, 🔥🔥🔥both dataset files downloaded and libraries imported in previous notebook are still available for this second notebook [video](https://youtu.be/YHFJj8-ik-8?t=1366)
	- use `ls` to check the dataset files out


---

## <mark style="background: #FFB8EBA6;">Radek's newsletter Jan 24 2023</mark> 

> I want to find the 20% that moves the needle by 80%. That is <mark style="background: #D2B3FFA6;">the only way to survive in the modern world</mark> .

> There are just too many capable actors out there on the hunt for our attention.

> But maybe that is all good. Maybe not getting distracted needs to be our daily struggle.

> Not only not getting distracted by Netflix, but also <mark style="background: #D2B3FFA6;">not getting distracted by another free course</mark> that we could passively consume.

​  [Project based learning](https://click.convertkit-mail2.com/8kudnlkkdmfoh0xrwn5fk/3ohphkhq8vpnmdsr/aHR0cHM6Ly9yYWRla29zbXVsc2tpLmNvbS9nb2luZy1mcm9tLW5vdC1iZWluZy1hYmxlLXRvLWNvZGUtdG8tZGVlcC1sZWFybmluZy1oZXJvLw==) where you devise the project and work on it end-to-end is <mark style="background: #FF5582A6;">the holy grail of learning</mark> .

> We all struggle, we all are human. But in the end two things matter:

-   efficient use of your time
-   investing your time and money where it matters, that is in yourself

---

## <mark style="background: #D2B3FFA6;">Radek's newsletter Jan 31 2023</mark> 

> Not accepting that you are human. That you are vulnerable and fallible.

> <mark style="background: #FF5582A6;">By not telling the complete story of who you are you deeply hurt yourself.</mark> 

> Yes, deleting a tweet after publishing it is 100x better than not tweeting it in the first place.

> But this behavior has its <mark style="background: #FF5582A6;">roots in a sense of shame</mark> . Along with elevated, <mark style="background: #FF5582A6;">unattainable expectations</mark> .

> UNATTAINABLE to you where you are right now in your journey.

> But <mark style="background: #D2B3FFA6;">by not embracing who you are right now you limit your opportunities to learn and to grow.</mark> 

> Our mind evolved to protect us from harm. This is the main reason why we feel so uncomfortable in new situations.

> But if we <mark style="background: #BBFABBA6;">put ourselves in those uncomfortable spots and give our mind the chance to observe that actually, nothing bad happens to us, and we do this repeatedly...</mark> 

> This is how we learn.

> You do not become better by sitting on the bench. You become better by playing the game.

> <mark style="background: #BBFABBA6;">Give yourself the chance. Go out onto the field and swing that bat even if it feels that you stand no chance.</mark> 

> For first 100 swings you do not. But on the 101 swing something magical is likely to happen.


---

## Radek's YT channel: Math in Machine Learning, exploring the cutting edge of research, and more!

[00:00](https://www.youtube.com/watch?v=bnpE_yGQyQA&t=0s) Intro
[01:41](https://www.youtube.com/watch?v=bnpE_yGQyQA&t=101s) What brings Sairam to Twitter? 
[03:09](https://www.youtube.com/watch?v=bnpE_yGQyQA&t=189s) Is engaging with the community helpful for research work? 
[05:49](https://www.youtube.com/watch?v=bnpE_yGQyQA&t=349s) How did Sairam get into Machine Learning?
[08:32](https://www.youtube.com/watch?v=bnpE_yGQyQA&t=512s) What is the role of math in achieving state-of-the-art results?
[10:03](https://www.youtube.com/watch?v=bnpE_yGQyQA&t=603s) Why are papers written the way they are (and the role of code in sharing ML ideas)
[13:49](https://www.youtube.com/watch?v=bnpE_yGQyQA&t=829s) How to get started with communicating online (Twitter, blogging, etc)
[16:11](https://www.youtube.com/watch?v=bnpE_yGQyQA&t=971s) What makes stable diffusion exciting?

<mark style="background: #FFB8EBA6;">Shining points</mark> 

- kaggle discussion can be a place where researchers look for new and promising ideas
- Coding provides a more natural way to achieve understanding of math
- It's ok to feel academic papers read like alien languages as more than often they are written with only a super minority audience in mind
- Sharing on twitter can sharpen your own ideas and understanding of things learnt