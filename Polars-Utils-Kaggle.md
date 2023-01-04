

<mark style="background: #FFB8EBA6;">RESOURCES</mark> 

divefastai : [kaggle otto a beginner's map](https://github.com/EmbraceLife/myfastaivault/blob/main/divefastai/00fastainbs/kaggle_otto_a_beginner's_map.md)

[LeaderBoard Ranking](https://www.kaggle.com/competitions/otto-recommender-system/leaderboard#)  [discussions](https://www.kaggle.com/competitions/otto-recommender-system/discussion?sort=votes)  [notebooks](https://www.kaggle.com/competitions/otto-recommender-system/code?competitionId=38760&sortBy=voteCount) Radek's [notebooks](https://www.kaggle.com/radek1/code) cdeotte's [notebooks](https://www.kaggle.com/cdeotte/code) OTTO dataset [repo](https://github.com/otto-de/recsys-dataset#dataset-statistics) my [notebooks](https://www.kaggle.com/danielliao/code?scroll=true), [chatgpt](https://chat.openai.com/chat)

[pandas](https://wesmckinney.com/book/pandas-basics.html#pandas_dataframe), [cuDF](https://docs.rapids.ai/api/cudf/stable/), [polars](https://pola-rs.github.io/polars-book/user-guide/quickstart/intro.html), [covisitation and candidate dataset](https://www.kaggle.com/datasets/danielliao/co-visiation-matrices-with-candidates-v1), fastai [streamlit](https://fmussari-fts-fastai-youtube-playlists-app-eohwrp.streamlit.app/), GPU for python [video](https://www.youtube.com/watch?v=5s8PljqLdkA)

Radek's [thread](https://twitter.com/radekosmulski/status/1597080115686805506) on how to succeed in otto

new ideas: a faster Candidate ReRanker [notebook](https://www.kaggle.com/code/adaubas/otto-fast-handcrafted-model), NN [model](https://www.kaggle.com/competitions/otto-recommender-system/discussion/370756#2056631) , [fastai](https://www.kaggle.com/code/shravankumar147/can-we-use-fastai), handcraft [improve](https://www.kaggle.com/code/tuongkhang/otto-pipeline2-lb-0-576/comments#2049094), [great example on how to go from basic with baselines](https://www.kaggle.com/code/junjitakeshima/otto-easy-understanding-for-beginner-en/comments#2057777), build on [w2v](https://www.kaggle.com/code/alexandershumilin/otto-word2vec/comments#2048246), [EDA eg](https://www.kaggle.com/code/adaubas/otto-interesting-times-series-eda-on-products/notebook), the best [EDA](https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline) (turned into polars I should), second best [EDA](https://www.kaggle.com/code/cdeotte/time-series-eda-users-and-real-sessions) 

A kaggle GM on [youtube](https://twitter.com/Rob_Mulla ) 

---

<mark style="background: #FFB8EBA6;">MY UTILS</mark> 

- search emoj in obsidian `cmd + j` 
- load my utils as a lib onto kaggle [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113382947&cellId=6)
- how to allow multi-output in a cell [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113955605&cellId=2)
- how to <mark style="background: #ADCCFFA6;">expand column width</mark> of pandas dataframe [cell](https://www.kaggle.com/code/danielliao/peek-at-otto-jsonl-dataset?scriptVersionId=114894270&cellId=4)
- download kaggle dataset `kaggle competitions download -c otto-recommender-system`
- unzip `unzip otto-recommender-system.zip`
- run scripts in terminal 
	- `git clone github-repo`
	- `pip install pipenv`, `pipenv sync`, `pipenv install -dev` `pipenv install numpy`
	- `pipenv run python -m src.testset --train-set train.jsonl --days 7 --output-path 'out/' --seed 42 `
- upload local data files into kaggle datasets [how](https://github.com/Kaggle/kaggle-api/wiki/Dataset-Metadata ) from e.g., paperspace
	- `kaggle datasets init -p /path/to/dataset` to create metadata
	- `vim dataset-metadata.json` to update the name of the dataset
	- `kaggle datasets create` (create a new Dataset)
	- `kaggle datasets version` (create a new version for an existing Dataset)


---


<mark style="background: #FFB8EBA6;">Debugging with `return`, `pp`</mark> 
user [guide](https://github.com/alexmojaki/snoop) on `pp`
😱 😂 🚀 how to debugging to understand each line of a script: example [notebook](https://www.kaggle.com/danielliao/evaluate-otto-organizer-script)  ^3ac7a9
- what does `submission.csv` look like? debugging `predictions = f.readlines()[1:]` see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115225853&cellId=6)
- what does `test_labels` look like by debugging `labels = f.readlines()`, see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115249537&cellId=4)
- what does each `label` in `for label in tqdm(labels, desc="Preparing labels"):` look like, see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115249817&cellId=4)
- what does `final_labels` from `prepare_labels(labels)` look like? see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115254505&cellId=6)
- what does `sid_type, preds = prediction.strip().split(",")` in `def prepare_predictions(predictions):` look like? see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115253508&cellId=7)
- how each `prediction` is converted to `prepared_predictions` in `def prepare_predictions(predictions):`, see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115254231&cellId=7)
- what does `def num_events(labels, k: int):` do,  see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115255061&cellId=6)
- what does `evaluate_session` do for each session, see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115260419&cellId=7); also make the debugging super fast by using subset `labels = f.readlines()[:10] # add [:1000] for fast run`, see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115260419&cellId=6)
- what does `evaluate_sessions` do for all sessions when a session is predicted, see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115258151&cellId=7); when a session is not predicted, see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115258151&cellId=8);
- what does `click_hits`, `cart_hits` is None mean, see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115260419&cellId=7)
- what to do when a session is not predicted, see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115258151&cellId=8) 
- what does `def recall_by_event_type(evalutated_events: dict, total_number_events: dict):` do? see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115274325&cellId=7)
- what does `def weighted_recalls(recalls: dict, weights: dict):` do? see [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115275248&cellId=7)




---
**<mark style="background: #FFB8EBA6;">POLARS</mark>** 

- How to print `head` and `tail` together with `suffix` [cell](https://www.kaggle.com/code/danielliao/reimplement-otto-train-validation-in-polars?scriptVersionId=114980240&cellId=30)
- check `width`, `height`, `shape[0]` of a df, [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115390387&cellId=18)
- 

<mark style="background: #FFB86CA6;">How to create dataframe or series</mark> 
- how to convert a dict of dicts into a list of dicts? [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115365367&cellId=17)
- How to create a dataframe with a list of dicts with `pl.DataFrame`? [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115365367&cellId=18)
- How to create a dataframe from a dict of lists? [api](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.from_dict.html#polars-from-dict)
- How to create a dataframe from a list of dicts with `from_dicts`? [api](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.from_dicts.html), [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115365367&cellId=19)
- how to save dictionary into a json file? [cell](https://www.kaggle.com/code/danielliao/evaluate-otto-organizer-script?scriptVersionId=115358721&cellId=12)
- how to read parquet file? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113386816&cellId=33)
- how to scan parquet file with super speed? [cell](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint?scriptVersionId=114185126&cellId=5)
- how to just read a 100 rows of data from a huge json file with `pl.scan_ndjson`? [cell](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint?scriptVersionId=114180650&cellId=26)
	- why do we need it given `fetch`, because if the large is too large, `fetch` can blow out 30GB RAM

<mark style="background: #FFB86CA6;">How to deal with `None` or `null`</mark> 
- How to create 3 `null` and `append` to a column? [api](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.rechunk.html#polars.Expr.rechunk)
- how to check `None` with `is_null` `is_not_null` and create `null` with `pl.lit(None)`? [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115343714&cellId=21)
- how many `NA`s or `null`s in every column? [cell1](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115389565&cellId=15), [cell2](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115389565&cellId=16) 

<mark style="background: #FFB86CA6;">Whether an element is in a list with `is_in`</mark> 
- example: ` pl.col('ground_truth').arr.first().is_in(pl.col('labels')).cast(pl.Int8).alias('click_hits_1')`  [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115343714&cellId=21)

<mark style="background: #FFB86CA6;">How to do set, Counter</mark> 
- combine `set` and `Counter` by `value_counts` in polars. [api](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.value_counts.html) [video](https://youtu.be/VHqn7ufiilE?t=611)

<mark style="background: #FFB86CA6;">How to work with groupby</mark> 
- how to groupby [video](https://youtu.be/VHqn7ufiilE?t=475) 

<mark style="background: #FFB86CA6;">How to work with datetime</mark> 
- how to filter between two datetimes? [video](https://youtu.be/VHqn7ufiilE?t=410)


<mark style="background: #FFB86CA6;">How to change dtypes in polars</mark> 
- `pl.Uint8` vs `pl.Int8` [cell](https://www.kaggle.com/code/danielliao/compare-train-test-full-with-radek?scriptVersionId=115164243&cellId=5), and convert from `Int8` to `UInt8` in [cell2](https://www.kaggle.com/code/danielliao/compare-train-test-full-with-radek?scriptVersionId=115164243&cellId=7)

<mark style="background: #FFB86CA6;">How to read or split a huge dataframe into chunks</mark> 
- How to split a large dataframe into multiples dataframes based on groups with `partition_by`? [api](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.partition_by.html)

<mark style="background: #FFB86CA6;">How to do merge, join, concat in polars</mark> 
- melt in polars [api](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.melt.html)
- how to `join` with `left_on` and `right_on`? [video](https://youtu.be/VHqn7ufiilE?t=554) 
- how to `join` two dataframes which share a single column? [api](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.join.html#polars.DataFrame.join), [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115090928&cellId=30)
- how to `concat` two dfs? [video](https://youtu.be/VHqn7ufiilE?t=577)
- how to `join` 2 dfs `on` two columns `on=['session', 'type']` and by `how='outer'` look like? [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115343714&cellId=20)
- how to `join` 3 dfs like this `click_hits.join(cart_hits, on='session').join(order_hits, on='session').sort('session')`? [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115355042&cellId=40)

<mark style="background: #FFB86CA6;">How two dataframes work together</mark> 
- How to do ops on two cols from two different dataframes? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115141923&cellId=76)
- Are their differences between two sum columns all zero? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115141923&cellId=76) 

<mark style="background: #FFB86CA6;">How to select or filter rows</mark> 
- How to use `slice`? [api](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.slice.html)
- How to use `shift` to remove the last event of each session? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115098705&cellId=25)
- How to select the unique rows when a column has lists instead of scalar value (`unique`, `is_unique`, `is_duplicated` can't be applied) ? `groupby`, `pl.all().first()` can work. [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115131052&cellId=49) 

<mark style="background: #FFB86CA6;">How random.seed work with polars</mark> 
- `random.seed(42)` and `random.randint` must run one by one or in the same cell, otherwise, rerun `random.randint` will generate something different. [notebook](https://www.kaggle.com/code/danielliao/compare-random-seed-between-kaggle-paperspace/)
- How to create random seed in polars? (actually not by polars, but `np.random.seed(42)`)  [example](https://pola-rs.github.io/polars-book/user-guide/dsl/expressions.html#expression-examples)
- When `random.seed` won't work in polars? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115144772&cellId=31)
- How `random.seed` should be used to work with polars? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115144772&cellId=29)
- discovered why that `random.randint(1,1)` can work, but `np.random.randint(1,1)` will fail? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115098705&cellId=28)


<mark style="background: #FFB86CA6;">How to chain expressions</mark> 
- How to chain every expression nicely? use `()` in the outer space [cell](https://www.kaggle.com/code/danielliao/reimplement-otto-train-validation-in-polars?scriptVersionId=114977157&cellId=27)
- When chain multiple `filter`s, we must use `collect` as early as possible to avoid [computer error](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-full-validation?scriptVersionId=115004300&cellId=41) and [RAM error](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-full-validation?scriptVersionId=115004300&cellId=40)? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-full-validation?scriptVersionId=115004300&cellId=43)

<mark style="background: #FFB86CA6;">How to config polars?</mark> 
- set num of rows to display? [api](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.Config.set_tbl_rows.html) [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115068991&cellId=24)
- set num of cols to display? [api](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.Config.set_tbl_cols.html)
- set the colwidth? [api](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.Config.set_fmt_str_lengths.html) [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115068991&cellId=24)
- reset the config to default? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115068991&cellId=24)
- how to set num of rows, cols, and colwidth in pandas? [guide](https://github.com/pola-rs/polars/issues/4547#issue-1348041739)

<mark style="background: #FFB86CA6;">How to check whether two dataframes are the same?</mark> 
- Two dataframes <mark style="background: #FF5582A6;">must have the columns sorted in the same way</mark>  with `sort`, before run `frame_equal` or `assert_frame_equal` [cell](https://www.kaggle.com/code/danielliao/compare-train-test-full-ms-with-cdeotte?scriptVersionId=115192421&cellId=12)
- when two dfs have a column with lists inside, <mark style="background: #FF5582A6;">the list needs to be sorted</mark> with `arr.sort` before `assert_frame_equal` [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115144772&cellId=80)
- How to compare series using `series_equal`? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115140482&cellId=64)
- use `testing.assert_frame_equal`, [api](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.testing.assert_frame_equal.html) and dataframes must be <mark style="background: #FF5582A6;">same type</mark> (lazy or not) [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-full-validation?scriptVersionId=115004300&cellId=49)
- use `testing.assert_series_equal`, [api](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.testing.assert_series_equal.html)
- use `frame_equal` to compare two dataframe [cell](https://www.kaggle.com/code/danielliao/reimplement-otto-train-validation-in-polars?scriptVersionId=114980731&cellId=32)
- check `n_unique` of each columns [cell](https://www.kaggle.com/code/danielliao/reimplement-otto-train-validation-in-polars?scriptVersionId=114980240&cellId=27)
- check the total rows with `count` [cell](https://www.kaggle.com/code/danielliao/reimplement-otto-train-validation-in-polars?scriptVersionId=114980240&cellId=27)
- check the `first`, `last`, `min`, `max` datetime [cell](https://www.kaggle.com/code/danielliao/reimplement-otto-train-validation-in-polars?scriptVersionId=114980240&cellId=26)

<mark style="background: #FFB86CA6;">How to select and deselect columns</mark> 
- how to `exclude` column(s) by name, wildcard and dtypes? [api](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.exclude.html)
- how to select all columns with `pl.all`? [api](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.all.html#polars.all)
- how to check bool values of a column to be True or not with `exp.all`? [api](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.all.html#polars.Expr.all) and [api](https://pola-rs.github.io/polars/py-polars/html/reference/series/api/polars.Series.all.html#polars.Series.all)

<mark style="background: #FFB86CA6;">How to work with `struct`</mark> 
- how to generate a `struct` by `value_counts`? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115387944&cellId=12)
- how to rename elements in the `struct` with `rename_fields`? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115387944&cellId=12)
- how to access the element of the `struct` by names with `struct.field`? [cell](https://www.kaggle.com/code/danielliao/eda-training-a-first-model-submission?scriptVersionId=115387944&cellId=12) 
- how to select two columns to use in one expression with `pl.struct(['col1', 'col2'])`? [api](https://pola-rs.github.io/polars-book/user-guide/dsl/custom_functions.html?highlight=pl.struct(%5B#combining-multiple-column-values), [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115343714&cellId=25)
- how to `split_exact` with `struct.rename_fields`, `to_frame`, `unnest`: [api](https://pola-rs.github.io/polars/py-polars/html/reference/series/api/polars.Series.str.split_exact.html), (spent much time but can't get it to work as in this  [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115288870&cellId=11))
- how to use two columns together? `pl.struct(['col1', 'col2'])`
- polars turn a list of dictionaries in a column into a list of structs.  [cell](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint?scriptVersionId=114178944&cellId=27)
- how to explode a list of structs above? [cell](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint?scriptVersionId=114178944&cellId=28), [cell](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint?scriptVersionId=114185126&cellId=31)
- how to split a `struct[3]` into 3 separate columns? [cell](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint?scriptVersionId=114178944&cellId=29)

<mark style="background: #FFB86CA6;">How to create new columns or overwrite a column</mark> 
- add new columns with `with_columns` [video](https://youtu.be/VHqn7ufiilE?t=475) 
- how to overwrite a column by using the same alias? [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115343714&cellId=21) 

<mark style="background: #FFB86CA6;">When if elif else needed</mark> 
- how to use if, elif, else or `when`, `then` `otherwise`? [api](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.when.html#polars.when) [cell](https://www.kaggle.com/code/danielliao/peek-at-otto-jsonl-dataset?scriptVersionId=114887553&cellId=12)

<mark style="background: #FFB86CA6;">When a filter is needed</mark> 
- how to do `filter` with `&` and `|` ? [api](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.filter.html#polars.DataFrame.filter), [video](https://youtu.be/VHqn7ufiilE?t=425)
- how to Series `filter`? [api](https://pola-rs.github.io/polars/py-polars/html/reference/series/api/polars.Series.filter.html#polars.Series.filter) 
- how to `filter` inside a context? [api](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.filter.html#polars.Expr.filter)
- how to <mark style="background: #FF5582A6;">save RAM with filter by using `&` and `|` instead of multiple `filter`s</mark> ? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-full-validation?scriptVersionId=114994639&cellId=33)
- but chain two `filter` s can produce different result from `filter` with `&` [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-full-validation?scriptVersionId=115004300&cellId=42)
- <mark style="background: #FF5582A6;">when chain multiple `filter`  s, we must stop being lazy as soon as possible with `collect`, so that RAM won't run out</mark> [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-full-validation?scriptVersionId=115004300&cellId=43)


<mark style="background: #FFB86CA6;">when every row of a column is a list</mark> 
- how to concat a list from one column with a list from another column with `pl.col('preds').arr.concat(pl.col('added_preds'))`? [cell](https://www.kaggle.com/code/danielliao/a-simple-pipeline?scriptVersionId=115146681&cellId=35)
- how to get the first 20 items of a list from a column with `arr.head(20`? [cell](https://www.kaggle.com/code/danielliao/a-simple-pipeline?scriptVersionId=115146681&cellId=35)
- how to cast every item of a list of a column into other types with `arr.eval(pl.element().cast(pl.Utf8))`? [cell](https://www.kaggle.com/code/danielliao/a-simple-pipeline?scriptVersionId=115146681&cellId=35)
- how to turn a column of lists of numbers into a column of lists of strings with `arr.eval(pl.element().cast(pl.Utf8))`, then into a long string separated by " " with `.arr.join(" ")`? [cell](https://www.kaggle.com/code/danielliao/a-simple-pipeline?scriptVersionId=115146681&cellId=44) 
- how to turn a single value into a list of a single value for each row of a column with `pl.concat_list(['label_clicks'])`? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115068991&cellId=25)
- can `unique` `is_unique`, `is_duplicated` help to get the unique rows when a column has a list for each row? No, they don't work on list. [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115131052&cellId=52)
- Let's count and compare the length of the list of 'ground_truth' column with `arr.lengths`, see whether the are the same length? (same) [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115141350&cellId=69), [cell2](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115141350&cellId=70)
- Why not compare the sum of the list between two `test_labels` dataframes with `arr.sum`? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115141923&cellId=75) 
- How to sort the lists of `ground_truth` columns with `arr.sort`? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115144772&cellId=80) Are they same after sorting the lists with `frame_equal`? [cell](https://www.kaggle.com/code/danielliao/reimplement-test-sessions-labels-validation?scriptVersionId=115144772&cellId=81)
- interesting example:  `pl.col('ground_truth').filter(pl.col('type') == 'clicks').arr.sum().count().alias('total_clicks'),` [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115300398&cellId=16)
- `pl.col('ground_truth').filter(pl.col('type') == 'clicks').arr.head(20).arr.lengths().sum().alias('total_clicks2'),`  [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115300398&cellId=16)
- `arr.unique`, `arr.lengths`, [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115343714&cellId=23)


<mark style="background: #FFB86CA6;">when it's a column of strings</mark> 
- how to `split` strings in a column into a list of substrings: [api](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/api/polars.Expr.str.split.html), [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115288870&cellId=11)


<mark style="background: #FFB86CA6;">doing statistics</mark> 
- how to do `describe` to df and series: [api-df](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/api/polars.DataFrame.describe.html#polars.DataFrame.describe), [api-series](https://pola-rs.github.io/polars/py-polars/html/reference/series/api/polars.Series.describe.html#polars.Series.describe), describe a series to see this [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115288870&cellId=14)

<mark style="background: #FFB86CA6;">map vs apply in polars</mark> 
- `map` (or `pl.duration` alike) vs `apply` (with `timedelta`) on speed and RAM usage [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113894779)
- how much more RAM is used by `apply` vs `map`
- how much slower is `apply` vs `map`
- how much faster is doing more parallel ops or making cols with `map`
- `apply` work with `set`, `intersection`, `len`, [cell](https://www.kaggle.com/code/danielliao/implement-evaluate-script-otto?scriptVersionId=115343714&cellId=25)
- `(pl.col('ts').max() - pl.col('ts').min()).apply(lambda x: timedelta(seconds=x))` [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113415551&cellId=45)


- how to convert validation-7-days jsonl files to parquet [notebook](https://www.kaggle.com/code/danielliao/recreate-validation-7-days-parquet?scriptVersionId=114747140)
- cold start on sessions/users between train (4 weeks) and test (1 week) sets? (YES, every session is new in test set) [notebook](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114729758&cellId=8)
- cold start on aids between train (4 weeks) and test (1 week) sets? (NO, no new aid in test set) [notebook](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114729919&cellId=8)
- cold start on sessions on Radek validation (train 3 weeks vs test 1 week)? Yes, 100% cold start on session [notebook](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114749591)
- cold start on aids on Radek validation (train 3 weeks vs test 1 week)? (2% new aids in validation test set from train set) [notebook](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114730145&cellId=9)
- 100% cold start problem on test sessions, and 0% cold start problem on test aids for newly created validation-7-days dataset [notebook](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114760325) 
- using `train_ms` to create `train_3_weeks` which match exactly with Radek's `train.parquet` [cell](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114679835&cellId=18) (not correct, because radek's validation is not correct)
- use `train_ms` to create `train_valid` in polars which could the correct see this version [notebook](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114818440)
- filter dataframe with a condition and `~` cell
- all about session length: `describe()` [cell](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114674102&cellId=20)
- use `random.seed` to create reproducibility? [cell](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114669415&cellId=8)
- find the largest, smallest and first datetime of a dataframe by converting timestamp in seconds to datetime? [cell](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114669415&cellId=15)
- add a column which tells which sessions whose first `ts` is greater than the `split_ts`? [cell](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114669415&cellId=17)
- select sessions whose first `ts` is greater than the `split_ts`? [cell](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114669415&cellId=17)
- convert the `split_ts` in seconds from a timestamp to datetime? [cell](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114669415&cellId=19)
- find out which or how many sessions whose length is just 1 in the entire train set? [cell](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114670007&cellId=21)
- there is no sessions whose length is just 1. [cell](https://www.kaggle.com/code/danielliao/reading-otto-recsys-organizer-scripts?scriptVersionId=114670007&cellId=23)
- how to turn a column of lists of numbers into a column of lists of strings, then into a long string separated by " "? [cell](https://www.kaggle.com/code/danielliao/kaggle-otto-pipeline-collections/#submission-format) 
- how to handle lists in a column? [cell](https://www.kaggle.com/code/danielliao/kaggle-otto-pipeline-collections/#submission-format)
- how to concat a list from one column with a list from another column? [cell](https://www.kaggle.com/code/danielliao/kaggle-otto-pipeline-collections/#submission-format)
- how to get the first 20 items of a list from a column? [cell](https://www.kaggle.com/code/danielliao/kaggle-otto-pipeline-collections/#submission-format)
- how to cast every item of a list of a column into other types? [cell](https://www.kaggle.com/code/danielliao/kaggle-otto-pipeline-collections/#submission-format)
- how to use str functions for strings of a column? [cell](https://www.kaggle.com/code/danielliao/kaggle-otto-pipeline-collections/#submission-format)



- 
- 
- how to cast columns from `int64` to `int32` and `int8`? [cell](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint?scriptVersionId=114178944&cellId=31)
- how to use `if, elif, else` in polars with `pl.when().then().otherwise(pl.when().then().otherwise())`? [cell](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint?scriptVersionId=114178944&cellId=39)
- how to experiment columns of a DataFrame? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113386816&cellId=36)
- how to subset 150000 sessions? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113386816&cellId=37)
- how to subset a single session of data? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113386816&cellId=39)
- how to experiment `max`, `min`, `count` on a column data? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113413374&cellId=44)
- how to insert a value from a pl.Expr into a function with `apply`? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113415551&cellId=45)
- how to groupby and run ops on cols within each group? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113428909&cellId=65) 
- how to add columns of the same value to a dataframe? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113457614&cellId=71)
- how to transform a row of a dataframe into a list with `transpose`, `to_series`, `to_list`? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113457614&cellId=74)
- on datetime, timestamp, duration with polars [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113893952)
	- convert from datetime to timestamp using `datetime` library
	- create a datetime object and convert datetime to timestamp and back and forth with `polars`
	- use `pl.duration` calculate duration and use `pl.Int64` and `dt.with_time_unit('ms')` to convert timestamp to datetime

- How to use `pl.duration` which has `map` in a situation where `groupby` context is needed? [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113896406)
	- `map` is used by `pl.duration` inferred from the error message and experiment afterward
	- use `groupby` to prepare inputs needed by `pl.duration` in a new dataframe
	- use `pl.duration` in the new dataframe
- How to use `apply` in polars [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113906602)
- how to turn a dataframe (only one row) into a list? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113912205&cellId=7)
- how to convert polars duration from seconds into hours? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113944778&cellId=7)
- when to use `fetch` vs `collect` notebook [notebook](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113956693)
- how to create an empty DataFrame? how to check whether a DataFrame is empty or not? [cell](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint?scriptVersionId=114178944&cellId=39)
- how to check the size or RAM used of a DataFrame? [cell](https://www.kaggle.com/code/danielliao/recreate-otto-full-optimized-memory-footprint?scriptVersionId=114178944&cellId=10)

**Techniques Experimented** #otto_edward 
- how to take only 15,000 lines from `train.jsonl` into a pandas df for analysis? [cell](https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline?scriptVersionId=112043205&cellId=10)
- how to run experiment to measure the speed of time one line of code or a block of code? [`%timeit`](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-time) or [`%%timeit`](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-timeit) -- not for result, only for speed testing [cells](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113382947&cellId=10)
- how to get a basic idea of time for running code? `%time` [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113382947&cellId=17)
- the most sure way to find out info of an object is with `help` [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113386816&cellId=23) 
- how to find out the difference between two timestamp in millisecond with `datetime.timedelta`? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113413374&cellId=42)
- how to check a long dict with a loop? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113457614&cellId=81)
- how to reorder a dict? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113563455&cellId=81)
	- sort by keys `dict(sorted(people.items()))`
	- sort by values `dict(sorted(people.items(), key=lambda item: item[1]))`
- how to find the most common/frequent items from a list with `Counter` and `most_common`? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113569269&cellId=104)
- how to check the size of an object with `sys.getsizeof`? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113675013&cellId=125)
- how to check the size of a dataframe with `memory_usage` and how to remove objects to save RAM with `del` and `gc.collect`? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113684713&cellId=16)
- how the RAM changes when loading libraries and datasets and removing objects [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113684713&cellId=21)
- how to read a jsonl file in `chunks` with `pd.read_json`?  [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113709846&cellId=26)  and what is the `chunks` object? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113709846&cellId=27)
- how to create features for each session info from a dataframe of multiple sessions in pandas [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113422864&cellId=62) and polars [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113428909&cellId=65)
- how to add feature columns addressing entire dataframe to the dataframe from above? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113457614&cellId=71)
- pandas
	- how to use tqdm in loops for reading file lines? [cell 3](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113373897&cellId=3)
	- how to use column `session` as index and remove `session` as column? [cell](https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline?scriptVersionId=112043205&cellId=11) 
	- how fast is `len(set(train.session))` vs `len(train.session.unique())`?  [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113382947&cellId=14)
	- how to prove that `len(set(train.session))` is much slower than `len(train.session.unique())`? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113709846&cellId=21)
	- differentiate `df[0]` vs `df.iloc[0]`, `df[:1]` vs `df.iloc[:1]`, `df.iloc[0].item()` vs `df.iloc[:1].item()` [cells](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113382947&cellId=21)
	- how to loop through each row of a dataframe with idx? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113420194&cellId=55)
	- how to use `tqdm` with loop of a dataframe? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113428909&cellId=60)
- seaborn plotting
	- how to draw barplot? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113457614&cellId=75) 
	- how to draw distributions? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113457614&cellId=78)
	- how to draw a vertical line as mean for the distribution? [cell](https://www.kaggle.com/code/danielliao/otto-getting-started-eda-baseline?scriptVersionId=113457614&cellId=78)


