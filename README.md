# Audio Classification Model
## Problem Statement
An audio classification deep learning model is essential for various applications that involve audio data, such as speech recognition, music genre classification, and audio event detection. The need for such a model arises from the fact that audio data is complex and often contains a vast amount of information that is not easily decipherable by human ears alone. For example, speech recognition systems require accurate and efficient audio classification to identify the speaker, recognize spoken words and phrases, and accurately transcribe them into text. Music genre classification systems also require audio classification to distinguish between different genres of music based on the audio characteristics of each song.
## Data Information
The dataset can be download using this [link](https://urbansounddataset.weebly.com/download-urbansound8k.html)

The dataset contains 8732 audio files of urban sound in WAV format. The sampling rate, bit depth, and number of channels are the same as those of the original file uploaded to Freesound. It also contains meta-data information about every audio file in the dataset. 
## Project Pipeline
* Understanding the Data:  We load the data into a dataframe using Pandas and understand the features present in it. This helps in choosing the features that will be needed for the final model.
* Data Preprocessing: Data preprocessing is the essential step of cleaning and transforming raw data to make it suitable for machine learning models. For the current dataset, we define a custom function for extracting the MFCC of each audio a file and create a new dataset. We also encode the target values using the LabelEncoder function available in sklearn library.
* Train/Test Split: The data is split into two sets: one for training the model and the other for testing its performance. We use the train_test_split function available in the sklearn library to perform the operation.
* Model Training and Evaluation: Here we can try different models until we get the desired level of performance on the given dataset. We use keras and Tensorflow to build the model. The model has 4 layers, the input and hidden layers having relu activation and the output layer having softmax activation. We also need to evaluate the models using appropriate evaluation metrics.
