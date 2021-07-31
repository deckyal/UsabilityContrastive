This folder contains emotion identity extractor software. 

In order to run this software you have to follow the procedure: 
  1. cd to the directory where requirements.txt is located.
  2. activate your virtualenv.
  3. run in your shell:
        pip install -r requirements.txt
  5. Go to to the folder res/models and follow the instructions to download and arrange the pretrained models.
  6. Go to Main.py and edit:
     1. the input video path
     2. the output directory path
  7. Run Main.py


Original the output will be placed into the /imgs folder.
For the emotion identity extractor software we used Python 3.6.
