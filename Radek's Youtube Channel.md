I have summarized Radek's video contents in terms of questions (answers provided)

<mark style="background: #FFB8EBA6;">2 Things I Learned on Kaggle Today</mark>  [video](https://youtu.be/S7pv_aU_ER8)
- 2 things for what? 
	- to substantially improve your public scores on Kaggle comp
- The 1st Thing, start [here](https://youtu.be/S7pv_aU_ER8?t=0)
	- what is the first technique?
		- adding labels for data sources by creating a new column
	- how does it work? 
		- if you are combining official dataset with external dataste for training
		- and if you add a column like `external` to distinguish official dataset from external dataset with labels `True` or `False`
		- your model will likely be trained with this new column added is likely to perform better
	- can it be applied to DL models? start [here](https://youtu.be/S7pv_aU_ER8?t=26)
		- yes, of course
	- why this new column can help model to infer better? start [here](https://youtu.be/S7pv_aU_ER8?t=37)
		- this column can enable model to try to understand how the differences of two datasets affect performance
		- the model can give more appropriate predictions on the test set based on which data source it is from.
- The 2nd Thing, start [here](https://youtu.be/S7pv_aU_ER8?t=59)
	- what's the 2nd technique?
		- adding a column (distance between your location and landmarks) to your dataset
	- how does it work?
		- if you have longtitude and lattitude columns for house locations
		- if you have landmarks geoinfo which enables the calculation of the distance between your locations and the landmarks
		- the distance column can very likely improve your model's performance
	- why?
		- because domain knowledge tells us that important landmarks can influence the price of houses a lot
- resources
	- 🎯 Add data source information: [https://www.kaggle.com/competitions/p...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbFRNeTJnY3FPTnJ3X2MtcWdNT2NlZG5CSXliQXxBQ3Jtc0trckJXZjdIdl8zNkFNX0EwUHQyUDM1RGt2WE8zbVAzVUZsTTY1X3VhMlRfVm9PUzVweURGbU54VlJlMkhHNDJ2RjNhR0hUZGMzdENUUFV5Zkc5dUM2UV9xWUlSejBrQ0dCb0EtWEc4VW9LVldKanRYRQ&q=https%3A%2F%2Fwww.kaggle.com%2Fcompetitions%2Fplayground-series-s3e1%2Fdiscussion%2F376043&v=S7pv_aU_ER8) 
	- 🗺️ Add distance from landmark feature: [https://www.kaggle.com/competitions/p...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbTRnRlFjWTRRZjBGa2dkek9JTjhnVVoxeng2QXxBQ3Jtc0tucVFWaHpMOEtsem5LN3R4UExZdElURVFqSHhDZVF4a0NrblZ1N0VRc01VVHNjRXUtcmNiTmNBRFhHR0N4WHNlMGVRSENVdGpjS2RtOFNzcVVSbFJlZklINExORkpvSDFIOXBQTm5oM28temd3UmI2cw&q=https%3A%2F%2Fwww.kaggle.com%2Fcompetitions%2Fplayground-series-s3e1%2Fdiscussion%2F376078&v=S7pv_aU_ER8) 
	- 🧨 Awesome notebook on Geospatial Feature Engineering and Visualization: [https://www.kaggle.com/code/camnugent...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbFkzdE92LXlXQ19DN2tPMlFMZDQxbXJHSGRoUXxBQ3Jtc0ttNW52blVfRERwaVR3THNaZExpUU1WOTktYmMyajRON2lNcEY2VmdhMDRrOFRoWmtqVV9HLXFIdzVuZmNTclcxdGJ1aG40STBSaWJYQW1ndW1HaS02eVRSYWxtSWtVRWljTTIyb2xIZDV4SVo5SENoRQ&q=https%3A%2F%2Fwww.kaggle.com%2Fcode%2Fcamnugent%2Fgeospatial-feature-engineering-and-visualization&v=S7pv_aU_ER8)