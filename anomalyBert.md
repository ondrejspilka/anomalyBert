Implement Python library and reference CLI of machine learning model which detects anomalies in timeseries data defined by timestamp and value. Result of detection is TOP N values alongside with their positions (timestamp informations). 
Architecturally it is implemented as encoder only transformer architecture which uses positional ancoding attention mechanism.
Propose parameters affecting number of layers, width of layers and depth of the network. Propose tokenization approach and how to generate embeddings for timeseries data. Pretrained model doesn't require additional finetuning, but implement mechanism allowinf finetuning on head(s).
Training data are the same format timestamp, value and probability value defining how value attributes to anomaly alongside with anomaly tag.
It implements normalization using min-max and optionally other algorithms.
Implementation is not using GPU, only CPU for training as well as inference, implementation can use frameworks like SciKit, Keras, Tensorflow, but pick as low as possible. 
Implement tests using synthetic datasets. For synthetic datasets create many different scenarios of test data, so they are stored in CSV files. Randomize the data so each generation produces different data. Cardinality of test data is from 10s to 100s of samples.  For the test datasets also create CLI commands to gerenate and store them. CLI must contain all commands for training and inference.
Create README.MD describing how model works and its usage.
Create wrapper for the library so model is in ONNX format.