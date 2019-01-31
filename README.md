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
I am using Anaconda Environment for a much easier user experience compared to the Python Environment. You can use either Anaconda or Python (https://www.python.org/downloads/)

1) Install Anaconda Python Environment Manager from the following link:
		https://conda.io/docs/user-guide/install/index.html
		
2) Create a new Python environment and install the required packages using the pip/conda install command.

## 3. DATASET
1) The dataset used for this work is a small subset of the YFCC dataset which can be obtained from "Yahoo Webscope” and the web address http://webscope.sandbox.yahoo.com	

2) The Records pertaining to San Francisco region(for first two steps), and other regions like India, United States, United Kingdom, Europe ( for the final step) are obtained by setting their respective geographical coordinates in `RefineDataset.ipynb`

## 4. CODE Folder
This folder contains the ipython notebooks for each individual task:
1. Refining the dataset
   - `RefineDataset.ipynb` is used to clean the dataset and store it in a pandas DataFrame.
2. Textual metadata processing 
   - `Text_Metadata_Proc.ipynb` is used to eliminate unnecessary characters and stopwords, stem words into root form, and extract the Photo ID, User ID, Latitude and Longitude from the previous Dataframe. 
3. Geographical data clustering
   - `GeoClustering_Exemplar.ipynb` is used to cluster popular locations together using HDBSCAN algorithm.
4. Region wise trend estimation 
   - `Trend-Master File-fr any Region.ipynb` is used to estimate the trend of tourist arrival for a particular region.
5. Seasonal Trend Analysis
   - `Seasonal_SF.ipynb` is used to analyze the seasonal trend of tourist arrival for a particular region.


## 5. RESULTS Folder
This folder contains the results obtained from each step. This can be used to better visualise the output and gain a better understanding of what is really happening.


## 6. REFERENCES Folder
This folder contains some research papers which were used as references in the scope of this project.


## 7. IMPLEMENTATION
1) The first step, textual metadata processing is performed in `Text_Metadata_Proc.ipynb` where the filtered SanFrancisco records `Filtered1M.csv` is fed as input.

2) The threshold value for support to perform filtering has to be set within the notebook and filtered records are stored in a file named `TP_op1M.csv` which is the result of the first step.

3) This `TP_op1M.csv` is fed as input to second step which is the geographical data clustering and exemplar identification in the `GeoClustering_Exemplar.ipynb`. The paramaters for the clustering can be altered within the ipython notebook and the result of the clustering is displayed as images within the notebook. The geographical values for exemplars can also be viewed within the notebook.

4) The region wise trend estimation for various regions is the file `Trend-Master File-fr any Region.ipynb`. The path for the input file is can be altered for respective regions within the notebook.The results of the implementation is recorded and stored in the file `Trend Analysis_final-Sheet1.pdf` in Results -> TrendEstimation.

5) The Seasonal trend analysis is performed on the same SanFrancisco records which is fed as input in the first step. Any other city can also be given as input whose paths can be changed in the `Seasonal_SF.ipynb` notebook and output can be viewed as images within the notebook. The results of the implementation is recorded and stored in Results -> TrendEstimation -> SeasonalTrend file.
