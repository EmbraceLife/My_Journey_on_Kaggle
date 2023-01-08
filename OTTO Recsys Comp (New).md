<mark style="background: #FFB8EBA6;">MY PIPELINES </mark> 

- pipeline collections [notebook](https://www.kaggle.com/code/danielliao/kaggle-otto-pipeline-collections)
- **A basic pipeline** introduced by #otto_edward  with [my corrected version](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113569269&cellId=115) it can score 0.483 in BL and my polars implmentation on full dataset is [here](https://www.kaggle.com/code/danielliao/kaggle-otto-pipeline-collections?scriptVersionId=114464575) with 0.484 BL score

---

<mark style="background: #FFB8EBA6;">Milestone OTTO NOTEBOOKS</mark> 

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