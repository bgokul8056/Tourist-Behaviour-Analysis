# Tourist-Behaviour-Analysis

1. REQUIREMENTS
	1.1 Python 3.5
	1.2 NLTK 
	1.3 folium
	1.4 matplotlib
	1.5 numpy
	1.6 basemap
	1.7 jupyter notebook
	1.8 FreeFile Viewer
2. SETUP INSTRUCTIONS
	2.1 Install Anaconda Python Environment Manager from the following link:-
		https://conda.io/docs/user-guide/install/index.html
	2.2 Create a Python 3.5 environment and install the requirements using the pip install command
3.CONTENTS OF THE FOLDER
	3.1 IpythonNotebooks - Ipynb notebooks for refining the dataset, textual metadata processing , geographical data clustering, region wise trend estimation , seasonal analysis.
	3.2 Results - Intermediate results of the implementation for each step in their respective folders and also San Francisco, US , UK, India, AUS, Europe, Africa regions record used in the implementation.
4. DATASET
		4.1	The dataset used for this work is a subset of the YFCC dataset which can be obtained from "Yahoo Webscope” and the web address http://webscope.sandbox.yahoo.com	
		4.2 	Records for SanFrancisco (for first two steps), and regions like US, UK, India, Europe ( for the final step) is obtained by setting the respective geographical coordinates in "RefineDataset.ipynb"
5. IMPLEMENTATION
	5.1 The first step, textual metadata processing is in "Text_Metadata_Proc.ipynb" where the filtered SanFrancisco records "Filtered1M" is fed.
	5.2 The threshold support value for filtering has to be set within the notebook and filtered records will be stored in a file named "TP_op1M.csv" which is the result of the first step.
	5.3 This "TP_op1M.csv" is fed as input to second step which is the geographical data clustering and exemplar identification in the "GeoClustering_Exemplar.ipynb". The paramaters for the clustering can be altered within the ipython notebook and the result of the clustering is displayed as images within the notebook. The geographical values for exemplars can also be viewed within the notebook.
	5.4 The region wise trend estimation for various regions is the file "Trend-Master File-fr any Region.ipynb". The path for the input file is can be altered for respective regions within the notebook.The results of the implementation is recorded and stored in the file "Trend Analysis_final-Sheet1.pdf" in Results -> TrendEstimation.
	5.5 The Seasonal trend analysis is performed on the same SanFrancisco records which is fed as input in the first step. Any other city can also be given as input whose paths can be changed in the "Seasonal_SF.ipynb" notebook and output can be viewed as images within the notebook. The results of the implementation is recorded and stored in Results -> TrendEstimation -> SeasonalTrend file.
