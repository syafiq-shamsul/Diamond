# Diamond

## 1. Summary
THe project aim is to predict the price of the diamond based on the featureds given from the dataset. The dataset obtained from https://www.kaggle.com/datasets/shivam2503/diamonds

## 2. IDE and Framework
IDE - Spyder
Frameworks - TensorFlow Keras, Pandas, Numpy, Matplotlib, & Scikit-learn

## 3. Methodology

## Data Pipeline
The unrelated columns of features from the data set will be removed initially. The categorical features will be encoded before the data splitting and stadardizing. The data is split to 80:20 for train:validation, then 80:20 for the validation:test. The train,validation and test features will be standardized.

## Model Pipeline
The model is trained using a simple feedfoward neural network. Dropout is applied to prevent the model to overfit. The model structure is visualized as in figure below
![model_structure](https://user-images.githubusercontent.com/100821053/163701449-b80c9435-5182-4088-a5d0-53082caa4c75.png)

The model is trained with batch size of 16 and epochs of 40 resulting :
Training MAE = 839
Validation MAE = 592

The figure below will illustrate the loss and MAE training process.
![loss](https://user-images.githubusercontent.com/100821053/163701510-2fdd872d-f957-42ce-932b-9054884ddc67.png)
![mae](https://user-images.githubusercontent.com/100821053/163701511-d0229836-772b-417d-baeb-5e05ab34bca9.png)

## 4. Results

The model is tested using the test data and resulting the MAE and MSE as shown in figure below:
![test result](https://user-images.githubusercontent.com/100821053/163701514-9c01da93-3f73-467b-898d-0c3dc82f8906.png)

The prediction is made with the test fata and is tabulated as in figure below:
![Predictions](https://user-images.githubusercontent.com/100821053/163701534-5a59d8fb-a187-4e87-828a-991a01416955.png)

Overall the results shows a positive trendline indicate the results as the labels with several outliers present in the graph.




