### Download Models

The pretrained affwildnet model for Valence and Arousal estimation comes from: [link](https://github.com/dkollias/Aff-Wild-models) <br>
The pretrained model for discrete emotion estimation comes from: [link](https://github.com/atulapra/Emotion-detection) <br>



### Folder Structure

After downloading the models you have to put them inside this directory to create a folder structure like:

    .
    ├───affwildnet
    │   ├───affwildnet-resnet-gru
    │   │   └───...
    │   ├───vggface
    │   │   └───...
    │   └───vggface_rnn
    │       └───...
    └───EmotionDetector
        └───model.h5
    
    
    
    C:\USERS\INTERNET\DESKTOP\GITHUB_PROJECT_UPLOAD\USABILITYCONTRASTIVE\EMOTIONIDENTITYEXTRACTOR\RES\MODELS
    ├───affwildnet
    │   ├───affwildnet-resnet-gru
    │   │   ├───resnet
    │   │   └───__MACOSX
    │   │       └───resnet
    │   ├───vggface
    │   │   ├───best_model_4096x2000x2
    │   │   │   ├───4096x2000x2
    │   │   │   └───__MACOSX
    │   │   │       └───4096x2000x2
    │   │   └───best_model_4096x4096x2
    │   │       └───4096x4096x2
    │   └───vggface_rnn
    └───EmotionDetector
