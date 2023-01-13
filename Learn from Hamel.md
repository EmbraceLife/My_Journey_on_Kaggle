## <mark style="background: #FFB86CA6;">🔥🔥🔥 Lineapy intro by Hamel </mark> [tweet](https://twitter.com/HamelHusain/status/1612989477622009856)

^5b7b1c

- `!pip install lineapy`
- `%load_ext lineapy`: must before everything else
- `import lineapy`
- if anything (df, int, dict, model, graph, etc) you want to<mark style="background: #BBFABBA6;"> track both values and pipelines</mark> along the way
- `art = lineapy.save(variable_to_track, 'lineapy_artifact_name')`, [video](https://youtu.be/2HpF3b-mM_4?t=677), save everything about this var up to this point
- after changes, run the same line above again, is to <mark style="background: #BBFABBA6;">save everything about this var up to this point</mark> 
- `art.get_value()` returns value of the var
- `art.get_code()` return only necessary codes to produce the var
- `art.get_session_code()` return all codes before the var
- `list_arts = lineapy.artifact_store()`, [video](https://youtu.be/2HpF3b-mM_4?t=1030), list all artifacts and their versions
- `lineapy.get("artifact_name", version=version_num)`, [video](https://youtu.be/2HpF3b-mM_4?t=1273), by default, latest version is chosen ^cce0f2
- `lineapy.delete("artifact_name", version=1)`, `version='latest'` or `version='all'`
- `linear.get_function(['cleaned_data_housing'], input_parameters=['data_url'])`, see [cell](https://www.kaggle.com/code/danielliao/lineapy-house-price?scriptVersionId=116085785&cellId=54)
- `repeat_eval_helper = lineapy.get_function(["accuracy","cv_scores","cv_accuracy_mean","cv_accuracy_std"], input_parameters=["modelname"], reuse_pre_computed_artifacts=["X_train", "X_test", "y_train", "y_test", "models"])` , var `modelname` is not tracked by lineapy in printed code
-  <mark style="background: #BBFABBA6;">use case</mark> : [discover_and_trace_past_work](https://github.com/LineaLabs/lineapy/blob/main/examples/use_cases/discover_and_trace_past_work/discover_and_trace_past_work.ipynb) 
- <mark style="background: #BBFABBA6;">use case</mark> : reuse dataframes, functions again for different models, [notebook](https://www.kaggle.com/danielliao/lineapy-reuse-component/), `lineapy.get_function` a more complex example
- <mark style="background: #BBFABBA6;">example</mark> : house price, [notebook](https://www.kaggle.com/danielliao/lineapy-house-price/) , [lineapy.save](https://www.kaggle.com/code/danielliao/lineapy-house-price?scriptVersionId=116085785&cellId=38) to save, [lineapy.get_function](https://www.kaggle.com/code/danielliao/lineapy-house-price?scriptVersionId=116085785&cellId=54) to act on new data