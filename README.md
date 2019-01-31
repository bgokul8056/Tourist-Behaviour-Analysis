# Tourist Behaviour Analysis

## 1. REQUIREMENTS
This project was done in Python 3.5 .The following Python packages are required in order to run this program. FreeFile Viewer ( Or any other similar software) can be used to initally have a glimpse of the dataset. You can also use Anaconda to make is easier to install the required packages.
####
	 
	 Natural Language Tool Kit (NLTK)
	 folium
	 matplotlib
	 numpy
	 basemap
	 jupyter notebook

## 2. SETUP INSTRUCTIONS
I am using Anaconda Python Environment for a much easier user experience compared to Python Environment. You can use either Anaconda or Python (https://www.python.org/downloads/)

1) Install Anaconda Python Environment Manager from the following link:
		https://conda.io/docs/user-guide/install/index.html
		
2) Create a Python 3.5 environment and install the required packages using the pip/conda install command.

## 3. CONTENTS OF THE CODE FOLDER
1) IpythonNotebooks - Ipynb notebooks for refining the dataset, textual metadata processing , geographical data clustering, region wise trend estimation , seasonal analysis.

2) Results - Intermediate results of the implementation for each step in their respective folders and also San Francisco, US , UK, India, AUS, Europe, Africa regions record used in the implementation.

## 4. RESULTS FOLDER
This folder contains the results obtained from each step. This can be used to better visualise the output and gain a better understanding of what is really happening.

## 5. REFERENCES FOLDER
This folder contains some research papers which were used as references in the scope of this project.

## 6. DATASET
4.1	The dataset used for this work is a subset of the YFCC dataset which can be obtained from "Yahoo Webscope” and the web address http://webscope.sandbox.yahoo.com	

4.2 	Records for San Francisco region(for first two steps), and other regions like India, United States, United Kingdom, Europe ( for the final step) is obtained by setting the respective geographical coordinates in "RefineDataset.ipynb"

## 5. IMPLEMENTATION
5.1 The first step, textual metadata processing is performed in `Text_Metadata_Proc.ipynb` where the filtered SanFrancisco records `Filtered1M.csv` is fed as input.

5.2 The threshold value for support to perform filtering has to be set within the notebook and filtered records are stored in a file named `TP_op1M.csv` which is the result of the first step.

5.3 This `TP_op1M.csv` is fed as input to second step which is the geographical data clustering and exemplar identification in the `GeoClustering_Exemplar.ipynb`. The paramaters for the clustering can be altered within the ipython notebook and the result of the clustering is displayed as images within the notebook. The geographical values for exemplars can also be viewed within the notebook.

5.4 The region wise trend estimation for various regions is the file `Trend-Master File-fr any Region.ipynb`. The path for the input file is can be altered for respective regions within the notebook.The results of the implementation is recorded and stored in the file `Trend Analysis_final-Sheet1.pdf` in Results -> TrendEstimation.

5.5 The Seasonal trend analysis is performed on the same SanFrancisco records which is fed as input in the first step. Any other city can also be given as input whose paths can be changed in the `Seasonal_SF.ipynb` notebook and output can be viewed as images within the notebook. The results of the implementation is recorded and stored in Results -> TrendEstimation -> SeasonalTrend file.
