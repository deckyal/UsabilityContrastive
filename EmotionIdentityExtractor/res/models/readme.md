The pretrained affwildnet model for Valence and Arousal estimation comes from: [link](https://github.com/dkollias/Aff-Wild-models)
The pretrained model for discrete emotion estimation comes from: [link](https://github.com/atulapra/Emotion-detection)

After downloading the models you have to put them inside this directory to create a folder structure like:

.
|
+--EmotionDetector
|  +--model.h5
|
+--affwildnet
|  +--affwildnet-resnet-gru
|  |  +-- ...
|  |  
|  +--vggface
|  |  +-- ...
|  |  
|  +--vggface_rnn
|  |  +-- ...
|  |  
|  +--.gitignore 
|
+--readme.md
