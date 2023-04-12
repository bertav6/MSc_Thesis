Code for MSc Thesis Design of an IT system based on Deep Learning for EEG artifact detection with BrainCapture

by Berta Viñas Redondo (s202256)

0_baseline: contains the code to evaluate the XGBoost, EEGNet and DSCNN models on the TUAR dataset, as binary and categorical classifications.
2_evaluate_on_bc: contains the code to evaluate the previous models on the BrainCapture recordings.
3_transfer_learning_bc: contains the code to apply transfer learning to the previous models and evaluate them on the BrainCapture recordings using LOSOCV.
6_computational_analysis: contains the code to transform TensorFlow models to TensorFlow Lite and evaluate their computational cost and time.
8_evaluate_unnanotated_bc: contains the code to visually evaluate the models on unnanotated BrainCapture recording
models: contains the defined models architecture and the models trained in this thesis.

data_preprocessing.py
load_data.py
model_evaluation.py
-> provide the functions need to load the datasets, preprocess and evaluate the models.

Notice that to run this code, the datasets from TUAR and BrainCapture recordings are needed.