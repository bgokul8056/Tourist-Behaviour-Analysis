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
1) The dataset used for this work is a small subset of the YFCC dataset which can be obtained from "Yahoo Webscope” (http://webscope.sandbox.yahoo.com)	

2) The Records pertaining to the San Francisco region and other regions like India, United States, United Kingdom, Europe are obtained by setting their respective geographical coordinates in `RefineDataset.ipynb`

## 4. TASKS Folder
This folder contains the ipython notebooks for each individual task:
1. Refining the dataset
   - `RefineDataset.ipynb` is used to set the geographical coordinates to extract data from a particular region.
2. Textual metadata processing 
   - `Text_Metadata_Proc.ipynb` is used to eliminate unnecessary characters and stopwords, and to stem words into their root form.
3. Geographical data clustering
   - `GeoClustering_Exemplar.ipynb` is used to cluster popular locations together using HDBSCAN algorithm.(https://hdbscan.readthedocs.io/en/latest/index.html)
4. Region wise trend estimation 
   - `Trend-Master File-fr any Region.ipynb` is used to estimate the trend of tourist arrival for a particular region.
5. Seasonal Trend Analysis
   - `Seasonal_SF.ipynb` is used to analyze the seasonal trend of tourist arrival for a particular region.


## 5. IMPLEMENTATION
1) The first step, "Textual metadata processing" is performed in `Text_Metadata_Proc.ipynb` where the filtered San Francisco records `Filtered1M.csv` is fed as input.

2) The threshold value for support to perform filtering has to be set within the notebook and the filtered records are stored in the file  `TP_op1M.csv`.

3) This `TP_op1M.csv` is now fed as input to the second step,"Geographical data clustering" in `GeoClustering_Exemplar.ipynb`. The paramaters for the clustering can be altered in the ipython notebook and the result of the clustering is displayed as images within the notebook. The geographical values for exemplars can also be viewed within the notebook.

4) "Region wise trend estimation" for various regions is performed in `Trend-Master File-fr any Region.ipynb`. The results of the implementation is recorded and stored in the file `Trend Analysis.pdf` in Results -> TrendEstimation.

5) "Seasonal trend analysis" is performed on the same San Francisco records which was fed as input to the first step. Any other region can also be given as input to `Seasonal_SF.ipynb` notebook and the output can be viewed as images in the notebook. The results of the implementation is recorded and stored in Results -> TrendEstimation -> SeasonalTrend file.

The following image shows the architecture of the proposed system.

![System Architecture](https://github.com/bgokul8056/Tourist-Behaviour-Analysis/blob/master/Results/arch_img.PNG)
        
      


## 6. RESULTS Folder
This folder contains the results obtained from each step. This can be used to better visualise the output and gain a better understanding of what is really happening.

1)InputRecords - Contains the `Filtered1M.csv` which is given as input for the Textual metadata processing step. `Filtered1M.csv` is obtained by extracting 1 million records from the YFCC Dataset using a python script.

2)TextualMetadataProcessing - Contains `TP_op1M.csv` which is the output of Textual metadata processing.

3)GeographicalDataClustering - Contains images which show cluster formation.

4)TrendEstimation - Contains several CSV files for several regions for which the trend has to be estimated. An analysis was performed on these regions and has been documented in `Trend Analysis.pdf`. Another folder titled 'SeasonalTrend' contains the seasonal trend analysis of the San Francisco region.
