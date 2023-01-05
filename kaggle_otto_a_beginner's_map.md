---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

**MY GOAL IN THIS COMPETITION**

**Goal in 3 steps**
- STEP 1 - understanding the problem including the dataset
- STEP 2 - building the baseline or simplest pipelines
- STEP 3 - gradually more tweaks on the pipelines

**Strategy**
- experimenting and remixing great notebooks from a beginner's perspective

**Why**
- one of my goals is described in a [reply](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline/comments#2060453) and inspired by a [discussion](https://www.kaggle.com/code/junjitakeshima/otto-easy-understanding-for-beginner-en/comments#2057637)
- [benefits](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline/comments#2060489) of dissecting and remixing great notebooks for a beginner (@radek1)


---

**GREAT INSIGHTS SHARED ON DISCUSSION**

**On Test Set**
- not appeared in train set, but could appear in earlier period and returning customers with existing carts and wishlist [source](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113569269&cellId=115)

---

**GREAT NOTEBOOKS TO BE STUDIED**
- OTTO - Getting Started (EDA + Baseline) notebook [original](https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline) by #otto_edward [mine](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113675013) notebook by #otto_dcl  #otto_dataset_organizer 

---

**HOW IS MY REPLICATE DIFFERENT FROM THE ORIGINAL NOTEBOOK**
- I experimented every line and extracted learning points which are useful and new to a beginner like myself in the sections below.
- Thanks to @radek1 for introducing polars library to us, I have implemented almost every line of the original notebook in polars.

---

**STAY FOCUS**
- techniques to do EDA with polars in particular
- understanding the problem and dataset
- build up the simple pipeline with polars and start experimenting

---

**Good Habits to learn**
- make using DEBUG a habit whenever I start building a notebook, otherwise it takes too long to run experiment later on. 
- del variables whenever a section of a notebook is finished to save RAM, otherwise it's easy to run out of memory later on. [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113675013&cellId=93)
- one version of the notebook, deal with one investigation, code from scratch, write only the necessaries

---


**TRIVIAL TYPOS FOUND IN THE ORIGINAL NOTEBOOKS** #otto_dcl  

- on the use of tqdm with `total` [discussion](https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline/comments?scriptVersionId=112043205#2060530)
- on the most frequent aids [disucssion](https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline/comments?scriptVersionId=112043205#2062309) (not that trivial )
- on the predictions generation error, see [discussion](https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline/comments#2063452), see cells for investigation  (not that trivial, as [it contributes 0.001 up](https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline/comments#2064677) in LB score)

---
**KNOW YOUR TOOLS**
- User [guides](https://pola-rs.github.io/polars-book/user-guide/dsl/folds.html), [APIs](https://pola-rs.github.io/polars/py-polars/html/reference/index.html) 
- docs of polars in a [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113756622&cellId=9)
- presentations of polars [2021.12](https://github.com/pola-rs/presentations/blob/main/mlconference.ai_08-12-2021/pres.ipynb)
- for pandas user [tutorial](https://pola-rs.github.io/polars-book/user-guide/coming_from_pandas.html)

---

**KNOW YOUR PROBLEM** #otto_edward 


- overview of the problem [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113457614&cellId=4)
- what does this competition want us to predict exactly? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113563455&cellId=93)
- how different are `clicks`, `carts`, `orders` are evaluated in the score? 1: 3: 6 [link](https://github.com/otto-de/recsys-dataset#evaluation) 

---




---

**KNOW YOUR DATA**


**On Datasets**
#otto_dataset_organizer_dataset : data [description](https://www.kaggle.com/competitions/otto-recommender-system/data)
#otto_radek_optimized_dataset: data [description](https://www.kaggle.com/competitions/otto-recommender-system/discussion/363843)
#otto_radek_optimized_polars: created by me from [notebook](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint/notebook?scriptVersionId=114178944) converting 11 GB into 1.2 GB on Kaggle, I have [checked](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint?scriptVersionId=114180650&cellId=4) the sameness

earlier exploration prior to #otto_radek_optimized_polars 
- load competition organizer's `train.jsonl` file (11 GB) and basic analysis with polars without blowing up 30GB RAM? [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113756622)
	- `pl.read_json` and `pl.read_ndjson` won't help, but `pl.scan_ndjson` and `.collect()` can do the trick (1 min) [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113756622&cellId=11) 
	- how to prove there is no duplicated sessions? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113756622&cellId=12)
	- now I can use polars to load and transform the otto `train.jsonl` to `train.parquet` [notebook](https://www.kaggle.com/code/danielliao/kaggle-otto-process-data?scriptVersionId=114116322)

---

---
<mark style="background: #FFB8EBA6;">Q&A </mark> 

- what the `density` stats on otto dataset is about? answered [here](https://github.com/otto-de/recsys-dataset/issues/2) <mark style="background: #ADCCFFA6;">answered</mark> 

---
<mark style="background: #FFB8EBA6;">MY PIPELINES </mark> 

- pipeline collections [notebook](https://www.kaggle.com/code/danielliao/kaggle-otto-pipeline-collections)
- **A basic pipeline** introduced by #otto_edward  with [my corrected version](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113569269&cellId=115) it can score 0.483 in BL and my polars implmentation on full dataset is [here](https://www.kaggle.com/code/danielliao/kaggle-otto-pipeline-collections?scriptVersionId=114464575) with 0.484 BL score

---
<mark style="background: #FFB8EBA6;">MY OTTO NOTEBOOKS</mark> 

- Explore otto full dataset (original in jsonl format) [notebook](https://www.kaggle.com/code/danielliao/peek-at-otto-jsonl-dataset/notebook)
- 😱 😂 🚀 Convert otto full dataset from jsonl to parquet and optimized in polars <mark style="background: #ABF7F7A6;">using kaggle's 30GB RAM</mark> [notebook](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint)
- 😱 😂 🚀 Create otto validation set (jsonl, split by the last 7 days) from <mark style="background: #ABF7F7A6;">running organizer's script on Kaggle</mark> [notebook](https://www.kaggle.com/code/danielliao/otto-organizer-script-on-kaggle?scriptVersionId=114850294) [validation-by-script-on-kaggle](https://www.kaggle.com/datasets/danielliao/otto-validation-7days-jsonl-from-script-on-kaggle), ([validation-set-1](https://www.kaggle.com/datasets/danielliao/my-valid-7day), [validation-set-2](https://www.kaggle.com/datasets/danielliao/validation-7days-otto-2) created using script on paperspace) 
- Optimize and Convert otto validation set from jsonl (<mark style="background: #ABF7F7A6;">generated by organizer's script on paperspace</mark> ) to parquet in polars using kaggle's 30GB RAM [notebook-1](https://www.kaggle.com/code/danielliao/recreate-validation-7-days-parquet?scriptVersionId=114747140) ([validation-7days-parquet](https://www.kaggle.com/datasets/danielliao/ottovalidation7days)), [notebook-2](https://www.kaggle.com/code/danielliao/recreate-otto-validation-7days-2nd?scriptVersionId=114816443) ([validation-7days-2nd-parquet](https://www.kaggle.com/datasets/danielliao/ottovalidation7days2nd))
- Optimize and Convert otto validation set (except <mark style="background: #FF5582A6;">test_labels</mark> ) from jsonl (<mark style="background: #ABF7F7A6;">generated on Kaggle</mark> ) to parquet in polars on Kaggle  [notebook-3](https://www.kaggle.com/code/danielliao/otto-validation-optimized-jsonl2parquet?scriptVersionId=114887627) ([validation-optimized-parquet](https://www.kaggle.com/datasets/danielliao/otto-validation-optimized-parquet))
- 😱 😂 🚀 Optimize and convert otto validation set (<mark style="background: #ABF7F7A6;">full, including test_labels</mark> ) from jsonl to parquet on Kaggle with polars  [experiment](https://www.kaggle.com/code/danielliao/peek-at-otto-jsonl-dataset#Let's-peek-at-test_labels.jsonl), [notebook](https://www.kaggle.com/code/danielliao/otto-validation-optimized-jsonl2parquet?scriptVersionId=114894810) for optimization and conversion, (created the [new optimized validation dataset](https://www.kaggle.com/datasets/danielliao/otto-validation-optimized-parquet) )
- 🚀 😂 🌟The [Discovery](https://twitter.com/shendusuipian/status/1607645668386164736) of a corruption of a validation set created by a Grandmaster and [conversations](https://www.kaggle.com/datasets/radek1/otto-train-and-test-data-for-local-validation/discussion/374405#2077900) with them
	- finding out which validation set has no cold start problem on aid, comparing validation from @radek1 and validations from mine [notebook](https://www.kaggle.com/danielliao/no-cold-start-aid-in-validation/) 

<mark style="background: #FFB8EBA6;">Notebooks to Reimplement Organizer's script</mark> 
 
- 😂 🚀 reimplement organizer's script in polars to create `train_sessions` or `train_valid` in otto validation set and verify its validity in this [notebook](https://www.kaggle.com/danielliao/reimplement-otto-train-validation-in-polars)
- 😱 😂 🚀 ⭐ reimplement organizer's script in polars to create `test_valid_full` or `test_sessions_full` and verify its validaty in this [notebook](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-full-validation?scriptVersionId=115004300) [story](https://forums.fast.ai/t/a-beginners-attempt-at-otto-with-a-focus-on-polars/102803/7?u=daniel)
- 😱 😂 🚀 reimplement `test_sessions` and `test_labels` and verify its validaty [script](https://github.com/otto-de/recsys-dataset/blob/main/src/testset.py#L34) , [notebook](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation), [story](https://forums.fast.ai/t/a-beginners-attempt-at-otto-with-a-focus-on-polars/102803/9?u=daniel),  [story-continued-2](https://forums.fast.ai/t/a-beginners-attempt-at-otto-with-a-focus-on-polars/102803/10?u=daniel)
- 😱 😂 🚀 reimplement organizer's `evaluate.py` script on kaggle: [notebook](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto)
	- run organizer's `evaluate.py` [script](https://github.com/otto-de/recsys-dataset/blob/0aa8346e0caec260ebd1cb47f556147cda5f770d/src/evaluate.py) on kaggle, using the evaluate [code](https://www.kaggle.com/danielliao/evaluate-otto-organizer-script/) in a pipeline [notebook](https://www.kaggle.com/danielliao/simple-pipeline-otto-1/) <mark style="background: #ADCCFFA6;">Done!</mark> 
	- 😱 😂 🚀 how to debugging to understand each line of the script above: [notebook](https://www.kaggle.com/danielliao/evaluate-otto-organizer-script) and story [[#^3ac7a9|inplace]] or [forum](https://forums.fast.ai/t/a-beginners-attempt-at-otto-with-a-focus-on-polars/102803/15?u=daniel) <mark style="background: #ADCCFFA6;">Done!</mark> 
	- 😱 😂 🚀 implement the script above in polars
		- implement `prepare_labels` and `prepare_predictions`, see [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115288870&cellId=6) <mark style="background: #ADCCFFA6;">Done!</mark> 
		- implement `num_events(labels, k)`, see [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115300398&cellId=16), confirmed by this [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115301417&cellId=7) <mark style="background: #ADCCFFA6;">Done!</mark> 
		- implement  `evaluate_session` and `evaluate_sessions`, `evaluated_events`, check script here [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115301417&cellId=9)  <mark style="background: #ADCCFFA6;">Done!</mark> 
			- implement `click_hits`, [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115343714&cellId=22)<mark style="background: #ADCCFFA6;">Done!</mark> 
			- implement `cart_hits`, [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115343714&cellId=25) <mark style="background: #ADCCFFA6;">Done!</mark> 
			- implement `order_hits`, [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115355042&cellId=35) <mark style="background: #ADCCFFA6;">Done!</mark> 
			- join them together, [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115355042&cellId=40) <mark style="background: #ADCCFFA6;">Done!</mark> 
			- to confirm my implementation result is the same to the organizer's result, [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115377521&cellId=41) <mark style="background: #ADCCFFA6;">Done!</mark> 
		- implement `recall_by_event_type` and `weighted_recalls`, check script in [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115378747&cellId=46) , and implemented [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115380231&cellId=49), confirmed [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115301417&cellId=8) <mark style="background: #ADCCFFA6;">Done!</mark> 
- 😱 using reimplementation notebooks above to split any subset of `train` into `train_sessions`, `test_sessions` and `test_labels` for fast experimentation on training and evaluating <mark style="background: #BBFABBA6;">Todo</mark> 
	- integrate my implementations together
	- 😱  Radek's [a-robust-local-validation-framework](https://www.kaggle.com/code/radek1/a-robust-local-validation-framework)  does subset, modeling, and evaluate in one go, let me reimplement it in polars

<mark style="background: #FFB8EBA6;">Notebooks to Verify My Dataset</mark> 

Are my handmade `train`, `test` of full dataset, and `train_sessions`, `test_sessions_full`, `test_sessions`, `test_labels`  of validation set the same to the ones generated by organizer's script?
-  😂 ⭐ Compare my `train.parquet` and `test.parquet`  from my [otto-radek-style-polars](https://www.kaggle.com/datasets/danielliao/otto-radek-style-polars) with Radek's `train` and `test` from [otto-full-optimized-memory-footprint](https://www.kaggle.com/datasets/radek1/otto-full-optimized-memory-footprint): <mark style="background: #ADCCFFA6;">Done</mark> ! experiment [notebook](https://www.kaggle.com/danielliao/compare-train-test-full-with-radek) (proved the same)
- 😂 ⭐ Compare my `train_ms.parquet` and `test_ms.parquet` with those from Colum2131's [otto-chunk-data-inparquet-format ](https://www.kaggle.com/datasets/columbia2131/otto-chunk-data-inparquet-format) (need [processing](https://www.kaggle.com/code/cdeotte/compute-validation-score-cv-565?scriptVersionId=111214251&cellId=5)): <mark style="background: #ADCCFFA6;">Done!</mark> (Same)  [notebook](https://www.kaggle.com/danielliao/compare-train-test-full-ms-with-cdeotte) 
- 😂 ⭐ Compare my `train_sessions` and `test_sessions_full` with those of [validation-7days-parquet](https://www.kaggle.com/datasets/danielliao/ottovalidation7days), [validation-7days-2nd-parquet](https://www.kaggle.com/datasets/danielliao/ottovalidation7days2nd), [new optimized validation dataset](https://www.kaggle.com/datasets/danielliao/otto-validation-optimized-parquet): <mark style="background: #ADCCFFA6;">Done!</mark> (Same! but radek's train is in different length, due to his using of old script) [notebook](https://www.kaggle.com/danielliao/compare-train-test-full-validation/)
- 😂 ⭐ Compare my `test_sessions` and `test_labels` with those of 3rd [dataset](https://www.kaggle.com/datasets/danielliao/otto-validation-optimized-parquet) and 4th validation sets (jsonl [dataset](https://www.kaggle.com/datasets/danielliao/otto-validation-4th-jsonl) and [notebook](https://www.kaggle.com/code/danielliao/4th-validation-set-jsonl?scriptVersionId=115160947), optimized parquet [dataset](https://www.kaggle.com/datasets/danielliao/validation-4th-optimized-parquet) and [notebook](https://www.kaggle.com/danielliao/4th-otto-validation-optimized-jsonl2parquet)), (both 3rd and 4th validation sets are made on Kaggle): <mark style="background: #ADCCFFA6;">Done!</mark> (Same) [notebook](https://www.kaggle.com/code/danielliao/compare-test-and-labels-validation/)
- 😂 ⭐ Compare my  `test_sessions` and `test_labels` with those of 1st validation set ([notebook](https://www.kaggle.com/danielliao/1st-otto-validation-optimized-jsonl2parque/), optimized parquet [dataset](https://www.kaggle.com/datasets/danielliao/validation-optimized-parquet-1st)) and 2nd validation set ([notebook](https://www.kaggle.com/danielliao/2nd-otto-validation-optimized-jsonl2parque/) and optimized parquet [dataset](https://www.kaggle.com/datasets/danielliao/otto-validation-optimized-parquet-2nd)): <mark style="background: #ADCCFFA6;">Done!</mark> (Same) [notebook](https://www.kaggle.com/danielliao/compare-test-and-labels-validation-1st2nd)
- 😂 ⭐ Compare 5th validation set (jsonl [datast](https://www.kaggle.com/datasets/danielliao/otto-validation-jsonl5th) created on paperspace without pipenv, [notebook](https://www.kaggle.com/danielliao/5th-otto-validation-optimized-jsonl2parque/) to create optimized-parquet [dataset](https://www.kaggle.com/datasets/danielliao/otto-validation-optimized-parquet-5th) on Kaggle) with 4th validation set: <mark style="background: #ADCCFFA6;">Done!</mark> (validation 1st, 2nd, 5th are the same as their jsonls are created on paperspace, even when 5th is created without pipenv ) [notebook](https://www.kaggle.com/code/danielliao/compare-test-and-labels-valid-4vs5), [story](https://forums.fast.ai/t/a-beginners-attempt-at-otto-with-a-focus-on-polars/102803/13?u=daniel)


<mark style="background: #FFB8EBA6;">Datasets Safe and Easy to Use</mark> 

- otto-train-set-test-set-optimized (both seconds and milliseconds, generated purely on Kaggle): [otto-radek-style-polars](https://www.kaggle.com/datasets/danielliao/otto-radek-style-polars)
- otto-validation-split-7-days (generated purely on Kaggle): [validation-4th-optimized-parquet](https://www.kaggle.com/datasets/danielliao/validation-4th-optimized-parquet)
---

<mark style="background: #FFB8EBA6;">ChatGPT helps coding</mark> 

- help make comments to codes I don't fully understand [video](https://youtu.be/NcCNw_UXnOc?t=66)
- 

---


---


**On first session** #otto_dcl
- create the following columns of a single session with polars -  [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113755135)
	- total_num_of_rows
	- total_num_unique_aids
	- total_num_of_clicks
	- total_num_of_carts
	- total_num_of_orders
	- session_starting
	- session_ending
	- total_duration_hour

**On first session** #otto_edward 
- Take a look at the first session and the first action of the first session?  [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113382947&cellId=29) 
- what are `session` and `events`? what are included inside `events`? what are `aid`, `type`, and `ts` in Unix timestamp or milliseconds? [cell](https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline?scriptVersionId=112043205&cellId=12)
- what is the exact time duration of the first session of the train set? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113413374&cellId=43)
- what are the frequencies of 'clicks', 'carts', 'orders' of the first session with a `dict` and `dict.get`? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113413374&cellId=46) , with `pl.select` and `pl.filter`? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113416812&cellId=50)

---



**On all sessions** #otto_dcl 

- session features on **session, aid, clicks, carts, orders** with polars [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113884305)
	- total_num_of_rows_or_actions_overall
	- total_num_of_unique_sessions_overall
	- total_num_of_unique_aids_overall
	- total_actions_in_each_session
	- total_clicks_in_each_session
	- total_carts_in_each_session
	- total_orders_in_each_session
- session features on **datetime, timestamp, duration** with polars [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113908414)
	- starting_datetime_each_session
	- ending_datetime_each_session
	- duration_seconds_each_session
	- duration_seconds_datetime_each_session [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113944778&cellId=7) 
	- duration_hours_datetime_each_session
	- duration_hours_int_each_session
- how does a session ends? [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113909979)
	- sessions can end anytime from near starting datetime to near ending datetime
	- sessions can last between near 28 days duration and 0 second duration
	- sessions can end with clicks, carts or orders
- plot distributions with seaborn [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113944778)
	- on total_actions_in_each_session
	- on duration_hours_int_each_session


**My investigation 3** : convert jsonl to parquet on Kaggle 30GB CPU with polars

**On all sessions** #otto_edward 
- what's the frequency of clicks, carts, orders overall in a bar chart? [cell]([cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113457614&cellId=75) )
- what's the distribution of num of actions of a user does on otto site overall? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113457614&cellId=78)
- what's the distribution of the duration (in hours) a session? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113457614&cellId=78)
- what are the interesting findings from the two graphs above? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113521379&cellId=79)


---


**On all aids** #otto_dcl 
- total_num_unique_aids in train set [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113970828)
	- `fetch` is for fast debugging
	- `collect` accesses full data 
	- `select` is a good habit to have
- groupby aid to find features 
	- total_occurs_each_unique_aid, distribution and quantiles [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=114046774)
	- total_occurs_as_clicks_each_aid [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=114068766&cellId=6)
	- total_occurs_as_carts_each_aid
	- total_occurs_as_orders_each_aid
	- cart_over_click_ratio_each_aid  [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=114068766&cellId=9)
	- order_over_click_ratio_each_aid
	- most_occurred_carts_highest_cart_click_ratio [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=114072005&cellId=11)
	- total_occurs_each_aid_whose_total_occurs_lt_30 [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=114062362&cellId=13)
	- plot distribution of total_occurs_each_aid_whose_total_occurs_lt_30 [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=114062362&cellId=13)


**On all aids** #otto_edward 
- what's the distribution of aids whose occurrences are under 30 in both pandas and polars? [cells](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113563455&cellId=82)
- what's the 5 most common aids of the dataframe? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113563455&cellId=91)
- what's the most frequent aids (for clicks, carts or orders) among the first 450000 sessions? [cells](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113569269&cellId=111)





---


---



---

**A PIPELINE EXAMPLE** - Radek's pipeline example

`create datasets to train on and for evaluation -> train MF/word2vec -> create a covisitation matrix (separate for train with validation and for the full train set for submission) -> create features & diagnostic code (measure hit rate, recall) for ranking models -> train a ranking model -> create submission based on this output`

---

**TODOS** - build my own pipeline based on Radek
- STEP 0 -  [💡 What is the co-visiation matrix, really?](https://www.kaggle.com/competitions/otto-recommender-system/discussion/365358) and how Radek do with co-visitation matrices [notebook](https://www.kaggle.com/code/radek1/co-visitation-matrix-simplified-imprvd-logic)
- STEP 1 - Candidate ReRank -  my edited copy [Candidate ReRank](https://www.kaggle.com/code/danielliao/candidate-rerank-model-lb-0-575-4795f5?scriptVersionId=112173816), [CR CV](https://www.kaggle.com/code/danielliao/compute-validation-score-cv-565?scriptVersionId=112173858), get me between 45 - 300 rank)
- STEP 2 - DEBUG option - applied to @cdeotte candidate rerank CV notebook (done, see details [[kaggle_cdeotte_candidate_rerank#^a86f9a | Debug option]])
- STEP 3 - DeepDive on Candidate ReRank - [[kaggle_cdeotte_candidate_rerank|here]] and in the process [[00fastainbs/my_journey_on_kaggle/kaggle_otto_a_beginner's_map#^013e28 | a little tweak]] gave my ranking a big leap 
- STEP 4 - Diving deep with @cdeotte's Candidate ReRank CV notebook (my [copy](https://www.kaggle.com/code/danielliao/compute-validation-score-cv-565?scriptVersionId=112319284)): use candidates + features + validation + XGB model to train [[kaggle_cdeotte_candidate_rerank#^f03541|how_to]] 🔥 and [guide](https://www.kaggle.com/competitions/otto-recommender-system/discussion/370210)
	- no need to do random split on 3-week-train using Radek's code based on @cdeotte's guide [[kaggle_cdeotte_candidate_rerank#^3df8ba | here]]
	- use 3-week-train to create co-visitation matrices
	- use matrices to generate candidates for valid sessions (with the help of handcrafted rules to rank?)
	- Build a training dataframe with candidates, validation set and many more features for training XGB models
	- 3 XGB models for 3 targets (clicks, carts, orders)
- STEP 4-5 - More of Radek's notebooks
	- use Matrix Factorization or [train Word2Vec](https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission#Training-a-word2vec-model) to generate candidates/features [demo](https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission#BONUS:-How-to-use-word2vec-to-generate-candidates/features-for-training-a-2-stage-recommender) 
	- LGBMRanker  [Radek notebook](https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker), XGB Radek [notebook](https://www.kaggle.com/code/radek1/training-an-xgboost-ranker-on-the-gpu)
	- Cross validation @radek1 [notebook](https://www.kaggle.com/code/radek1/eda-a-look-at-the-data-training-splits?scriptVersionId=112534706) 
	- Ensemble 


---


**How I got to my current standing on the LB and how to improve going forward** - Radek's [post](https://www.kaggle.com/competitions/otto-recommender-system/discussion/368278) ^c4f601

- 🔥 [random split](https://www.kaggle.com/code/radek1/a-robust-local-validation-framework?scriptVersionId=112366704&cellId=6) on the train set into train and train_labels for training the LGBM Ranker model using Radek's local validation [notebook](https://www.kaggle.com/code/radek1/a-robust-local-validation-framework)
	- do I need to remove the intersection sessions in the train set? no, Radek didn't do it and there is no strong intuition for doing it.
- a little tweak on the LGBM Ranker model: training with `dart`, reduce estimates from 20 to 10, (to try: L1/L2 regularization, subsampling etc)
- Writing [more features](https://www.kaggle.com/competitions/otto-recommender-system/discussion/368278#2048479) for training, Radek provided a list of features with their importances to LGBMRanker model
- feed candidates into train data and add features to it question raised [[00fastainbs/my_journey_on_kaggle/kaggle_otto_a_beginner's_map#^c6dc7e|here]] 
- train 3 models for clicks, carts and orders
- 


---

**QUESTIONS** - Burning and essential ones

What this problem is about? 
- given full complete sessions, to predict incomplete sessions
- use train data to learn patterns or relations between users and items, then given a few actions of a user, to predict what users will click, cart and order 


Do train set and test set share users? Do they share aids? Train set has complete sessions, test set has incomplete sessions, do full sessions have special features?

🔥 Can NN models predict without us creating features?

How to make the most out of LGBM Ranker notebook? 

🔥 <mark style="background: #D2B3FFA6;">answered</mark> How to integrate candidates from co-visitation matrices or **word2vec** into LGBMRanker model or XGB model?  @cdeotte's guide [here](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575/comments#2048484) (more recent and full [guide](https://www.kaggle.com/competitions/otto-recommender-system/discussion/370210)) and answer [here](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575/comments#2049308) and to Radek I [asked](https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker/comments#2048455) and got a [demo](https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission?scriptVersionId=112522144&cellId=18) and [answers](https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission/comments#2051575) to my detailed implementations  ^c6dc7e

🔥 XGB or LGBMRanker only predict one value, is that really a problem? @radek's [comment](https://www.kaggle.com/code/radek1/training-an-xgboost-ranker-on-the-gpu/comments#2048273)

🔥 why the second step is MF/word2vec?  what does it do? how is it different from co-visitation matrix 


<mark style="background: #D2B3FFA6;">answered</mark> Does Radek's LGBM Ranker notebook have Candidate Generation as the first stage of a two stage pipeline? asked [here](https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker/comments#2048452) and answered [here](https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker/comments#2048455)

<mark style="background: #D2B3FFA6;">answered</mark> How does LGBM Ranker take advantages of co-visitation matrices when creating features? [[kaggle_radek_LGMBRanker#^70298b| my understanding here]]

<mark style="background: #D2B3FFA6;">answered</mark> What are the individual importance of features for training LGBMRanker model? Where are they from? Why should we care about the importance? [here](https://www.kaggle.com/competitions/otto-recommender-system/discussion/368278#2048479) 

<mark style="background: #D2B3FFA6;">answered</mark>  Are chunk files disrupt the integrity of sessions? Are 1/4 restriction disrupt session integrity? will disruption of session integrity omit unique aid pairs, which may affect performance?  --> all files by @cdeotte [keep session integrity](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575/comments#2046880) --> I [realized](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575/comments#2046914)merge before cut ensure all aid_x miss no unique pairs from its session

<mark style="background: #D2B3FFA6;">answered</mark>  Why set the weights from 1 to 4 with point-slope form? [answered](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575/comments#2044426) by @cdeotte, experimention has performance implications

<mark style="background: #D2B3FFA6;">answered</mark>  Why at first @cdeotte use top 40 aids, then top 20, and then top 15 for one and top 20 for another? [answered](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575/comments#2048026)

<mark style="background: #D2B3FFA6;">answered</mark> why the weights for clicks, carts and orders are 1: 6: 3 by @radek1 [comment](https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission/comments#2043207)

---

**EXPERIMENTS** - 

- True experiments can be frustrating!
- Better to start with the smallest dataset, and make sure the code without error!
- Then increase the dataste gradually, to see how to optimize the code to speed up!


<mark style="background: #ABF7F7A6;">DONE</mark> moved `df = df.loc[(df.aid_x >= PART*SIZE)&(df.aid_x < (PART+1)*SIZE)]` before the merge for pairs, it results in much lower performance. Why? doing so, lose many unique pairs, see the notebook experiment [version](https://www.kaggle.com/code/danielliao/compute-validation-score-cv-565?scriptVersionId=112283447)

<mark style="background: #ABF7F7A6;">DONE</mark> add 'type_y' to `drop_duplicates` in the first co-visitation matrix to see whether it improves on performance [the experiment notebook version](https://www.kaggle.com/code/danielliao/compute-validation-score-cv-565?scriptVersionId=112319284) and this idea comes from discussion [here](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575/comments#2044545) --> CV notebook get 0.5660, BL [notebook version](https://www.kaggle.com/code/danielliao/candidate-rerank-model-lb-0-575-4795f5?scriptVersionId=112324484) get 0.576 (**Rank 28** on 2022-11-28) ^013e28

<mark style="background: #FF5582A6;">YET</mark> train Radek's LGBM Ranker model with full training set to random split into train (given aids) and train_labels (hidden aids)


---

**NOTEBOOKS Analysis**

[[kaggle_cdeotte_candidate_rerank]]
[[kaggle_radek_LGMBRanker]]


---

**MY PIPELINE** based on Radek

- create co-visitation matrices or word2vect using 3-weeks-train data
- create candidates for each session of valid data --> feed each aid of a session to a co-visitation matrix to the candidates of this aid, do it to all co-visitation matrices
- build a dataframe using the candidates 
	- row --> one candidate per row 
	- columns --> session, candidate, in_cvm1, in_cvm2, in_cvm3, click_label, cart_label, order_label
- create features for the candidates (dataframe)
	- but no `ts` nor `type` info for candidates, no worry, we can create other features like similarity score for them by @radek1 [comment](https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission/comments#2043742) 

STEP 1 - use **Test and Train sets combined** to train **candidate generation models** like word2vec [notebook](https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission) 
STEP 2 - use models to generate candidates for train_4th_week (split into test and test_labels using organizer's script) 
STEP 3 - join candidate dataframe with validation set and create lots of features on it
STEP 4 - use this new dataframe to train a **prediction model** like XGB [notebook](https://www.kaggle.com/code/radek1/training-an-xgboost-ranker-on-the-gpu/notebook), with 3 versions, one for 'clicks', one for 'carts', one for 'orders'
STEP 4 - Validation: split [notebook](https://www.kaggle.com/code/radek1/eda-a-look-at-the-data-training-splits) the train_3_weeks into train_2_weeks (for training word2vec) and train_3rd_week (for training XGB) and train_4th_week (for validating XGB) according to @radek1
STEP 5 - use the 3 trained models to predict on Test set, to produce predictions on 'clicks', 'carts' and 'orders'
STEP 6 - merge 3 predictions to meet the submission format 
STEP 7 - repeat the steps above to create another submission using covisitation matrix or matrix factorization ...
STEP 8 - vote ensemble [notebook](https://www.kaggle.com/code/radek1/2-methods-how-to-ensemble-predictions) on submission files from different models (one produced from word2vect, one from co-visitation matrix [notebook](https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575#Candidate-ReRank-Model-using-Handcrafted-Rules), one from Matrix Factorization) interesting discussion on [ensemble](https://www.kaggle.com/competitions/otto-recommender-system/discussion/368170#2045736)

---
**MEMORY MANAGEMENT**



DASK 
- Also consider using DASK which works with both CPU Pandas or GPU RAPIDS cuDF to use multiple CPU/GPU and/or disk when needed to avoid all memory errors!  
- https://www.dask.org/get-started

---
 **SPEED WITH GPU** 

- All of the following code will take time to run since this data has around 13 million users and 2 million items. 
- I suggest using an accelerated dataframe library for all the following processing like Nvidia's RAPIDS cuDF [here](https://rapids.ai/) which uses GPU instead of CPU for accelerated speed !
- cuDF + Dask => https://docs.rapids.ai/api/cudf/stable/user_guide/10min.html
- Question: is polars faster than cuDF with GPU? ask Radek [here](https://www.kaggle.com/competitions/otto-recommender-system/discussion/368170#2056398)


---

**CDEOTTE'S CANDIDATE DATAFRAME AND FEATURES** 
[post](https://www.kaggle.com/competitions/otto-recommender-system/discussion/370210)


**What Train data for XGB models look like**

| session | aid | user features | item features | user-item interaction features | click target | cart target | order target |
| ------- | --- | ------------- | ------------- | ------------------------------ | ------------ | ----------- | ------------ |
|         |     |               |               |                                | 1 or 0       | 1 or 0      | 1 or 0             |


**6 STEPS TO BUILD DATAFRAME TO TRAIN GBT MODEL**

more [ideas](https://www.kaggle.com/competitions/otto-recommender-system/discussion/370210#2058486) to build onto this pipeline

**step 0 - what does [data](https://github.com/otto-de/recsys-dataset#data-format) and [predictions](https://github.com/otto-de/recsys-dataset#submission-format), labels or ground truth look like**

- [my reading](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts) of organizer scripts and organizer's graph
- what does ground truth look like and how it is [created](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=112844992&cellId=9)
	- one click aid
	- others are unique cart aids or order aids
- how the valid_test and valid_test_labels are [split](https://github.com/otto-de/recsys-dataset#traintest-split)

 
**step 1 - create candidate dataframe**  [notebook](https://www.kaggle.com/code/danielliao/candidate-covisitation-gpu) 

- 50-200 candidates per session (as likely correct predictions), in the end model will predict and help select 20 predictions
- use` valid_train` (3 weeks) + `valid_test` (1 week random split first half) to train 3 co-visitation matrix [[00fastainbs/kaggle-script-cdeotte-compute-validation-score#^232b27|code here]],  [notebook](https://www.kaggle.com/code/danielliao/compute-validation-score-cv-565/) 
- better to save the trained co-visitation [files](https://www.kaggle.com/code/danielliao/compute-validation-score-cv-565/data?scriptVersionId=112319284) into a dataset
- load co-visitation matrices from parquet files into dict `pqt_to_dict` [[00fastainbs/kaggle-script-cdeotte-compute-validation-score#^e28483|see code]] 
- generate 50 candidates for each session in valid_test [[00fastainbs/kaggle-script-cdeotte-compute-validation-score#^d995c7|see here]] ^84qx87
	- do we have to remove all the aids from valid set? (maybe not)
- use 3 co-visitation matrices to generate candidates 


--- 

**step 2 - create item features**   [notebook](https://www.kaggle.com/code/danielliao/feature-generation-covisitation/)

- [how](https://www.kaggle.com/competitions/otto-recommender-system/discussion/366474) to think of features and models (radek's [guide](https://www.kaggle.com/competitions/otto-recommender-system/discussion/368278#2056887))
	- If the features are designed correctly, the reranker (XGB/LGBM models) should always beat heuristics
		- like cdeotte's Candidate ReRank with handcrafted rules). [source](https://www.kaggle.com/competitions/otto-recommender-system/discussion/366474#2032984) 
	- one row per item  [source](https://www.kaggle.com/competitions/otto-recommender-system/discussion/366474#2032979)
		- no duplicates, must aggregate results for models as "An XGB model cannot aggregate multiple rows for the same item in its training"
- Using our **train data + valid data** (yes use test leak)
- Create features based on `aid` regardless `session` => [[00fastainbs/kaggle-script-radek-lgbm-ranker#^4e94a3|code]] ready to apply
	- `item_item_count` => group by aid, count rows, name column "item_num_occurrences_overall" => the higher occurrences overall, the more likely it is clicked/cart/buy
	- `item_user_count` => group by aid, count unqiue sessions, add column "item_num_users_know_it" => the more users (click, cart, order) it, the more likely it is clicked/cart/buy
	- `item_buy_ratio`   => group by aid, count rows when `type == orders`, divide by  `item_user_count` , add column "item_ratio_users_buy_over_know"=> the more people will buy it after knowing it, the more likely it's a cart or buy
	- `item_cart_ratio`   => count occurrences of an aid of type `carts` / `item_user_count` => the more people carted it after knowing it, the more likely it is a cart or buy
	- `add_action_num_reverse_chrono`: group by each session, cumcount the rows, descending order  [[00fastainbs/kaggle-script-radek-lgbm-ranker#^4e94a3|code]]
- we create item [features](https://www.kaggle.com/competitions/otto-recommender-system/discussion/368278#2048479) in **their own dataframe**, see Radek does it in LGMBRanker notebook
- split the dataframe and run in chunks and save a parquet to disk code => [[00fastainbs/kaggle-script-cdeotte-compute-validation-score#^64f42e|code]] is ready to use

```python
item_features = train.groupby('aid').agg({'aid':'count','session','nunique','type','mean'})
item_features.columns = ['item_item_count','item_user_count','item_buy_ratio']
# CONVERT COLUMNS TO INT32 and FLOAT32 HERE
item_features.to_parquet('item_features.pqt')




```

**NOTE**: How to deal with memory problem in this step? 

Reduce dtype
- Always reduce dtype to save RAM, like `df[col] = df[col].astype('int32')`` or even `int8`

Processing everything in chunks for user features
- If you are using `train.groupby('session')` to make user features, then perhaps first split the train data into X pieces (like 2, 4, 8 pieces).
- Then process user features for each piece separately and save to disk as `f'user_features_p{PIECE_NUMBER}.pqt'`.
- Later when you read them in, you can concatenate them together. (for a demo, see @cdeotte covisitation matrix code below)

Processing everything in chunks for item features
- break `train` into 10 dataframe parts. 
- All rows pertained to a single item must be in the same dataframe part so that `train_part_1.groupby('aid')` will work correctly. 
- After processing, save each part separately to disk. 
- And later when you read them from disk, concatenate them together before merging them to candidate dataframe.

---

**step 3 - create user features** [notebook](https://www.kaggle.com/code/danielliao/feature-generation-covisitation/)

- **Using our validation data**, we create user features in **their own dataframe** 
- and save a parquet to disk. For example
- Features
	- `user_item_count` => group by session, count rows, name column "num_items_viewed" => the higher occurrences overall, the more likely it is clicked/cart/buy
	- `user_item_clicked` => group by session, count unqiue sessions, add column "item_num_users_know_it" => the more users (click, cart, order) it, the more likely it is clicked/cart/buy
	- `user_buy_ratio`   => group by session, count rows when `type == orders`, divide by  `item_user_count` , add column "item_ratio_users_buy_over_know"=> the more people will buy it after knowing it, the more likely it's a cart or buy
	- `user_cart_ratio`   => count occurrences of an aid of type `carts` / `item_user_count` => the more people carted it after knowing it, the more likely it is a cart or buy
	- `add_action_num_reverse_chrono`: group by each session, cumcount the rows, descending order  [[00fastainbs/kaggle-script-radek-lgbm-ranker#^4e94a3|code]]

```python
user_features = train.groupby('session').agg({'session':'count','aid','nunique','type','mean'})
user_features.columns = ['user_user_count','user_item_count','user_buy_ratio']
# CONVERT COLUMNS TO INT32 and FLOAT32 HERE
user_features.to_parquet('user_features.pqt')
```

---

**step 4 - create user-item interaction features**

- **Using our validation data**, we create **multiple** user-item feature **dataframes** and save them as parquets to disk. 
- For **each idea**, we can make a **new dataframe**. 
	- why separate dataframes? 
- One **dataframe** can contain all items that **a user clicks**. 
	- So make a dataframe with one column `user`, one column `item`, and a third column called `item_clicked`. 
	- Then for each unique item that a user clicked, we add a new row with `item_clicked = 1`. 
	- Note that our dataframe will have **no duplicate rows** of `['user','item']` pairs. 
	- Save this dataframe to disk. 
	- When we merge this to candidate dataframe we will `fillna(0)` to indicate the items not clicked.

---
**Step 5 - Add features to our candidate dataframe** [notebook](https://www.kaggle.com/danielliao/merge-candidates-features/)

- To add features to our candidate dataframe, we read from disk and merge them on as follows

```python
item_features = pd.read_parquet('item_features.pqt')
candidates = candidates.merge(item_features, left_on='aid', right_index=True, how='left').fillna(-1)
user_features = pd.read_parquet('user_features.pqt')
candidates = candidates.merge(user_features, left_on='session', right_index=True, how='left').fillna(-1)
```

- Now our candidate dataframe looks like

| user | item | item_feat1 | item_feat2 | user_feat1 | user_feat2 |
| ---- | ---- | ---------- | ---------- | ---------- | ---------- |
| 0001 | 6456 | 10         | 12         | 3          | 0.5        |
| 0001 | 4490 | 13         | 5          | 5.4        | 0.1        |
| 0002 | 8486 | 55         | 10         | 5          | 0.9        |


- **NOTE**: when having memory problems **merging all our features to our candidate dataframe**, then we can do this in **chunks**

```python
CHUNKS = 10 # split into 10 chunks
chunk_size = len(candidates)/CHUNKS
for k in range(CHUNKS):
    df = candidates.iloc[k*chunk_size:(k+1)*chunk_size].copy()
    df = df.merge(item_features, left_on='aid', right_index=True, how='left').fillna(-1) # if aid matches, then add columns to all rows grouped by the aid
    df = df.merge(user_features, left_on='session', right_index=True, how='left').fillna(-1) # if session matches, then add columns to all rows grouped by the session
    df.to_parquet(f'candidate_with_features_p{k}.pqt')
```

---

**Step 6 - add targets to our candidate dataframe from step 1**

- The best way to add a column of targets is to use dataframe merge. 
- First we make a dataframe of all the `target=1` as follows. 
- Starting with a dataframe that contains the targets as a column of lists ( like Radek's ground truth labels [here](https://www.kaggle.com/datasets/radek1/otto-train-and-test-data-for-local-validation?select=test_labels.parquet)) such as:

`test_labels.parquet`

| session | type   | ground_truth |
| ------- | ------ | ------------ |
| 0001    | clicks | [3456, 4490, 5661, 7821, 9914 ]             |
| 0002    | clicks | [1222, 4656, 533, 8486]             |

We use the following code to convert these lists into a dataframe of targets:

```python
tar = pd.read_parquet('test_labels.parquet')
tar = tar.loc[ tar['type']=='clicks' ]
tar = tar.labels.explode().astype('int32')
tar.columns = ['user','item']
tar['click'] = 1
```

This produces a dataframe `click_target` like

| user | item | click |
| ---- | ---- | ----- |
| 0001 | 3456 | 1     |
| 0001 | 4490 | 1     |


And we merge it to our candidate dataframe with the following line:

```python
candidates = candidates.merge(click_target,on=['user','item'],how='left').fillna(0)
```

| user | item | item_feat1 | item_feat2 | user_feat1 | user_feat2 | click |
| ---- | ---- | ---------- | ---------- | ---------- | ---------- | ----- |
| 0001 | 6456 | 10         | 12         | 3          | 0.5        | 0     |
| 0001    | 4490 | 13         | 5          | 5.4        | 0.1        |  1     |
| 0002 | 8486 | 55         | 10         | 5          | 0.9        |      1 |

---

**TRAINING**
- how to think of XGB models, [disucssion](https://www.kaggle.com/competitions/otto-recommender-system/discussion/366477) 
- We now have train data for our GBT ranker model. 
- We must train using `GroupKFold`. 
- **Important Note**: when we train, we do not use the `user` and `item` columns as features, we only use the other columns. 
- `FEATURES = candidates.columns[2 : -1*len(targets)]`. 
- Note with XGB, we have 3 options for rankers by changing `objective` parameter, to either `rank:pairwise`, or `rank:ndcg`, or `rank:map`

```python
import xgboost as xgb
from sklearn.model_selection import GroupKFold

skf = GroupKFold(n_splits=5)
for fold,(train_idx, valid_idx) in enumerate(skf.split(candidates, candidates['click'], groups=candidates['user'] )):

    X_train = candidates.loc[train_idx, FEATURES]
    y_train = candidates.loc[train_idx, 'click']
    X_valid = candidates.loc[valid_idx, FEATURES]
    y_valid = candidates.loc[valid_idx, 'click']

    # IF YOU HAVE 50 CANDIDATE WE USE 50 BELOW
    dtrain = xgb.DMatrix(X_train, y_train, group=[50] * (len(train_idx)//50) ) 
    dvalid = xgb.DMatrix(X_valid, y_valid, group=[50] * (len(valid_idx)//50) ) 

    xgb_parms = {'objective':'rank:pairwise', 'tree_method':'gpu_hist'}
    model = xgb.train(xgb_parms, 
        dtrain=dtrain,
        evals=[(dtrain,'train'),(dvalid,'valid')],
        num_boost_round=1000,
        verbose_eval=100)
    model.save_model(f'XGB_fold{fold}_click.xgb') # 5 folds of data, results in 5 models for just clicks
```

- **NOTE** If you have memory problems training XGB on GPU, consider downsampling negatives 2x, 4x, 10x, 20x with `frac = 0.5, 0.25, 0.1, or 0.05`. Or use DASK XGB with multiple GPUs. Here is example code: 
- (@radek1 has done the similar in his notebook too)

```python
positives = candidates.loc[candidates['click']==1]
negatives = candidates.loc[candidates['click']==0].sample(frac=0.5)
candidates = pd.concat([positives,negatives],axis=0,ignore_index=True) # built for training models
```
---

**INFERENCE**

- For inference, we create a new candidate dataframe (using our techniques from before but this time from all 4 weeks of Kaggle train for co-visitation matrices). 
- Then we make item features from all 4 weeks of Kaggle train plus 1 week of Kaggle test. 
- And we make user features from Kaggle test. 
- We merge the features to our candidates. 
- Then we use our saved models to infer predictions for clicks. 
- Lastly we select 20 by sorting the predictions and choosing 20 with.

```PYTHON
preds = np.zeros(len(test_candidates))
for fold in range(5):
    model = xgb.Booster()
    model.load_model(f'XGB_fold{fold}_click.xgb')
    model.set_param({'predictor': 'gpu_predictor'})
    dtest = xgb.DMatrix(data=test_candidates[FEATURES])
    preds += model.predict(dtest)/5 # predict on each row of the candidate dataframe, each prediction is the average of predictions of 5 models
predictions = test_candidates[['user','item']].copy()
predictions['pred'] = preds

predictions = predictions.sort_values(['user','pred'], ascending=[True,False]).reset_index(drop=True)
predictions['n'] = predictions.groupby('user').item.cumcount().astype('int8') # use cumcount as index
predictions = predictions.loc[predictions.n<20] # select the top 20 predictions out of 50 candidates in each session
sub = predictions.groupby('user').item.apply(list)
sub = sub.to_frame().reset_index()
sub.item = sub.item.apply(lambda x: " ".join(map(str,x)))
sub.columns = ['session_type','labels']
sub.session_type = sub.session_type.astype('str')+ '_clicks'
```

- **NOTE** if you have memory errors. 
- Consider loading 1/10th of the test data. 
- Then merge features. 
- Then infer. 
- Next load the next 1/10th, 
- merge features, 
- infer. etc. etc. 
- Lastly concatenate the predictions and make submission.csv

Have fun! Enjoy!

---


```python


############################

def candidate_generation_w2v(aid, num_candidates): pass
def candidate_generation_cvm(aid, num_candidates): pass

df_valid = pl.read_parquet('../input/otto-train-and-test-data-for-local-validation/test.parquet') # train_4th_week
df_valid_labels = pl.read_parquet('../input/otto-train-and-test-data-for-local-validation/test_labels.parquet')

list_unique_sessions_valid = df_valid.session.unique().to_list()
list_of_lists_all_aids_of_each_session_valid = df_valid.groupby('session').agg(pl.col('aid').to_list())
list_of_df_candidates_of_all_sessions_valid = []
for idx, session_id in enumerate(list_unique_sessions_valid):
	list_of_all_aids_of_each_session = list_of_lists_all_aids_of_each_session_valid[idx]
	num_aids_each_session = len(list_of_all_aids_of_each_session)
	list_of_candidates_merged_each_session = []
	list_of_candidate_orders_merged_each_session = []
	list_of_candidate_orders_as_weight_merged_each_session = []
	for idx_aid, aid in enumerate(list_of_all_aids_of_each_session):
		if num_aids_each_session >= 20:
			num_candidates_to_generate = 0
		elif num_aids_each_session >= 10:
			num_candidates_to_generate = 2
		elif num_aids_each_session >= 5:
			num_candidates_to_generate = 3
		elif num_aids_each_session >= 2:
			num_candidates_to_generate = 10
		else: # num_aids_each_session == 1 (question: how many sessions in test data has just 1 row data)
			num_candidates_to_generate = 20			
		list_of_candidates_each_aid = candidate_generation_w2v(aid, num_candidates_to_generate)
		list_of_orders_for_candidates_of_each_aid = list(range(1, len(list_of_candidates_each_aid)+1))
		list_of_orders_as_weight_for_candidates_of_each_aid = list(range(1, len(list_of_candidates_each_aid)+1))[::-1]
		list_of_candidates_merged_each_session += list_of_candidates_each_aid
		list_of_candidate_orders_merged_each_session += list_of_orders_for_candidates_of_each_aid
		list_of_candidate_orders_as_weight_merged_each_session += list_of_orders_as_weight_for_candidates_of_each_aid
	df_candidates_lists_each_session = pl.DataFrame(['session': session_id, 'candidates': list_of_candidates_merged_each_session, 'candidate_order': list_of_candidate_orders_merged_each_session, 'candidate_order_as_weight': list_of_candidate_orders_as_weight_merged_each_session])
	df_candidates_explode_each_session = df_candidates_lists_each_session.explode(['candidate', 'candidate_order', 'candidate_order_as_weight'])
	df_candidates_explode_occur_each_session = df_candidates_explode_each_session.select([pl.col('*'),pl.col('aid').count().over(['session', 'aid']).alias('occur')])
	df_candidates_explode_occur_unique_each_session = df_candidates_explode_occur_each_session.drop_duplicates(['session', 'aid'])
	df_candidates_explode_occur_weight_unique_each_session = df_candidates_explode_occur_unique_each_session.select([pl.col('*'), pl.col('occur')*pl.col('candidate_order_as_weight').alias('weight')])
	df_candidates_explode_occur_weight_unique_each_session = df_candidates_explode_occur_weight_unique_each_session.sort_values(['session', 'weight'], ascending=[True, False])
	
	num_to_keep = 50 # according to cdeotte, the num for each session should be between 50-200
	df_candidates_selected_each_session = df_candidates_explode_occur_weight_unique_each_session.head(num_to_keep).over('session')
	list_of_df_candidates_of_all_sessions_valid.append(df_candidates_selected_each_session)

df_candidates_of_all_sessions_valid = pl.concat(list_of_df_candidates_of_all_sessions_valid)
df_valid_merged_candidates_all_sessions = df_valid.join(df_candidates_of_all_sessions_valid, on=['session', 'aid'], how='outer')

## add label columns (click, cart, order label columns) to df_valid_merged_candidates_all_sessions
type2id = {"clicks": 0, "carts": 1, "orders": 2}

df_valid_labels = df_valid_labels.explode('ground_truth').with_columns([
    pl.col('ground_truth').alias('aid'),
    pl.col('type').apply(lambda x: type2id[x])
])[['session', 'type', 'aid']]

df_valid_labels = df_valid_labels.with_columns([
    pl.col('session').cast(pl.datatypes.Int32),
    pl.col('type').cast(pl.datatypes.UInt8),
    pl.col('aid').cast(pl.datatypes.Int32)
])

df_valid_labels = df_valid_labels.with_column(pl.lit(1).alias('gt'))
df_valid_labels = df_valid_labels.with_columns(pl.col('gt').(if 'type' == 0, 'gt' == 1, else 'gt' == 0).alias('click_lable'))
df_valid_labels = df_valid_labels.with_columns(pl.col('gt').(if 'type' == 1, 'gt' == 1, else 'gt' == 0).alias('cart_lable'))
df_valid_labels = df_valid_labels.with_columns(pl.col('gt').(if 'type' == 2, 'gt' == 1, else 'gt' == 0).alias('order_lable'))
df_valid_merged_candidates_all_sessions_with_labels = df_valid_merged_candidates_all_sessions.join(df_valid_labels, how='left', on=['session', 'type', 'aid']).with_columns(pl.col('gt').fill_null(0), pl.col('click_label').fill_null(0), pl.col('cart_label').fill_null(0), pl.col('order_label').fill_null(0))

## create features =========================================
def add_action_num_reverse_chrono(df):
    return df.select([
        pl.col('*'),
        pl.col('session').cumcount().reverse().over('session').alias('action_num_reverse_chrono')
    ])

def add_session_length(df):
    return df.select([
        pl.col('*'),
        pl.col('session').count().over('session').alias('session_length')
    ])

def add_log_recency_score(df):
    linear_interpolation = 0.1 + ((1-0.1) / (df['session_length']-1)) * (df['session_length']-df['action_num_reverse_chrono']-1)
    return df.with_columns(pl.Series(2**linear_interpolation - 1).alias('log_recency_score')).fill_nan(1)

def add_type_weighted_log_recency_score(df):
    type_weights = {0:1, 1:6, 2:3}
    type_weighted_log_recency_score = pl.Series(df['type'].apply(lambda x: type_weights[x]) * df['log_recency_score'])
    return df.with_column(type_weighted_log_recency_score.alias('type_weighted_log_recency_score'))

def apply(df, pipeline):
    for f in pipeline:
        df = f(df)
    return df

pipeline = [add_action_num_reverse_chrono, add_session_length, add_log_recency_score, add_type_weighted_log_recency_score]

df_valid_merged_candidates_all_sessions_with_labels_features = apply(df_valid_merged_candidates_all_sessions_with_labels, pipeline)

## Running XGB model =========================================
from nvtabular import *
from merlin.schema.tags import Tags
import polars as pl
import xgboost as xgb

from merlin.core.utils import Distributed
from merlin.models.xgb import XGBoost
from nvtabular.ops import AddTags

train_ds = Dataset(df_valid_merged_candidates_all_sessions_with_labels_features.to_pandas())

feature_cols = ['aid', 'type','action_num_reverse_chrono', 'session_length', 'log_recency_score', 'type_weighted_log_recency_score']
target = ['gt'] >> AddTags([Tags.TARGET])
qid_column = ['session'] >>  AddTags([Tags.USER_ID]) # we will use sessions as a query ID column
                                                     # in XGBoost parlance this a way of grouping together for training
                                                     # when training with LGBM we had to calculate session lengths, but here the model does all the work for us!

wf = Workflow(feature_cols + target + qid_column)
train_processed = wf.fit_transform(train_ds)
ranker = XGBoost(train_processed.schema, objective='rank:pairwise')

# version mismatch doesn't result in a loss of functionality here for us
# it stems from the versions of libraries that the Kaggle vm comes preinstalled with

with Distributed():
    ranker.fit(train_processed)

test = pl.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')
test = apply(test, pipeline)
test_ds = Dataset(test.to_pandas())

wf = wf.remove_inputs(['gt']) # we don't have ground truth information in test!

test_ds_transformed = wf.transform(test_ds)
test_preds = ranker.booster.predict(xgb.DMatrix(test_ds_transformed.compute()))

# submission
test = test.with_columns(pl.Series(name='score', values=test_preds))
test_predictions = test.sort(['session', 'score'], reverse=True).groupby('session').agg([
    pl.col('aid').limit(20).list()
])

session_types = []
labels = []

for session, preds in zip(test_predictions['session'].to_numpy(), test_predictions['aid'].to_numpy()):
    l = ' '.join(str(p) for p in preds)
    for session_type in ['clicks', 'carts', 'orders']:
        labels.append(l)
        session_types.append(f'{session}_{session_type}')

submission = pl.DataFrame({'session_type': session_types, 'labels': labels})
submission.write_csv('submission.csv')
```


STEP 2 GENERATING CANDIDATES
- Create a large data for training
    -   Validation set (with ground truth)
    -   Candidates added to each session of validation set
    -   Turn ground truth column into 3 columns (clicks, carts, orders)
    -   Create many features
-   Create candidates for sessions
    -   Use all data (train set + test set) to train word2vec or covisitation to ensure all aids are seen by the models
    -   Use the models to generate candidates for each session of validation set
-   How many candidates to generate
    -   If sessions have more than 20 aids, no candidate needed
    -   If not, add more to fill up to 20 aids
-   How to select needed candidates from more candidates
    -   for each aid in a session
        -   Select 20 candidates
        -   Keep candidates orders in another list too
    -   Merge lists for entire session
        -   Merge lists of candidates
        -   Merge lists of orders
    -   Put these 2 lists into dataframe
    -   Create weight by occurrence and generation order
    -   Select the needed number according to each session using the weight
-   How to decide the num of candidates to generate
    -   If session has more than 20 aids, no candidates generation needed
    -   If less than 20 aids, more than 10 in a session, generate 2 candidates for each aid
    -   If aids is between (5,10), generate 3 for each aid
    -   If aids less than 5, generate 10 candidates for each aids
    -   If aids == 1, generate 20
-   If aids<20,select (20-aids) candidates
-   How to decide how many to select




---

candidate list from 3 co-visitation matrices

| aid_x_1 | aid_y_1 | aid_x_2 | aid_y_2 | aid_x_3 | aid_y_3 |
| ------- | ------- | ------- | ------- | ------- | ------- |
| 0       | [2,3]   | 0       |   [1,2]      | 0       | [3,4]   |
| 1       | [4,5,6] | 1       |   [3,4]      | 1       | [5,6]        |

If we add candidates to train data for training model, we need to build the following dataframe for training a model to predict clicks

| session_valid | aid_valid | cand-cvm1 | cvm1 | cvm2 | cvm3 | click_label | cart_label | order_label |
| ------------- | --------- | --------- | ---- | ---- | ---- | ----------- | ---------- | ----------- |
| 0             | 0         | [0,1,2,3] | 1    | 0    | 0    | 1           | 0          | 0           |
| 0             | 1         | [0,1,2,3] | 0    | 1    | 0    | 1           | 0          | 0           |
| 0             | 2         | [0,1,2,3] | 0    | 1    | 0    | 0           | 1          | 0           |
| 0             | 3         | [0,1,2,3] | 0    | 0    | 1    | 0           | 1          | 0           |
| 0             | 4         | [0,1,2,3] | 1    | 0    | 1    | 0           | 0          | 0           |
| 1             | 3         | [4,5,6,7] | 1    | 0    | 1    | 1           | 0          | 0           |
| 1             | 6         | [4,5,6,7] | 1    | 0    | 1    | 0           | 0          | 1           |
| 1             | 8         | [4,5,6,7] | 0    | 0    | 0    | 0           | 1          | 0           |
| 1             | 1         | [4,5,6,7] | 0    | 0    | 0    | 0           | 0          | 0           |

Earlier version

| session_valid | aid_valid | cand-cvm1 | cand-cvm2 | candidates_clicks | candidates_carts | cvm1 | cvm2 | cvm3 | click_label | cart_label | order_label |
| ------------- | --------- | --------------- | ---------------- | ----------------- | ---------------- | ---- | ---- | ---- | ----------- | ---------- | ----------- |
| 0             | 0          | [0,1,2,3]       | [0,0,1,1]        | 0                 | 0                | 1    | 0    | 0    | 1           | 0          | 0           |
| 0             | 1          | [0,1,2,3]       | [0,0,1,1]        | 1                 | 1                | 0    | 1    | 0    | 1           | 0          | 0           |
| 0             | 2          | [0,1,2,3]       | [0,0,1,1]        | 2                 | 2                | 0    | 1    | 0    | 0           | 1          | 0           |
| 0             | 3          | [0,1,2,3]       | [0,0,1,1]        | 3                 | 3                | 0    | 0    | 1    | 0           | 1          | 0           |
| 0             | 4          | [0,1,2,3]       | [0,0,1,1]        | 4                 | 4                | 1    | 0    | 1    | 0           | 0          | 0           |
| 1             |           | [4,5,6,7]       | [1,0,2,1]        | 5                 | 5                | 1    | 0    | 1    | 1           | 0          | 0           |
| 1             |           | [4,5,6,7]       | [1,0,2,1]        | 6                 | 6                | 1    | 0    | 1    | 0           | 0          | 1           |
| 1             |           | [4,5,6,7]       | [1,0,2,1]        | 7                 | 7                | 0    | 0    | 0    | 0           | 1          | 0           |
| 1             |           | [4,5,6,7]       | [1,0,2,1]        | 8                 | 8                | 0    | 0    | 0    | 0           | 0          | 0           |