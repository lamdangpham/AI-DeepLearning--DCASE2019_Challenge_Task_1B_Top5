# CNN For Task 1A/B DCASE2019 challenge
http://dcase.community/challenge2019/task-acoustic-scene-classification#subtask-a
http://dcase.community/challenge2019/task-acoustic-scene-classification#subtask-b

# Hierarchy:
- 01_Technique_Report: Technique report for DCASE2019 Challenge - task 1A/B
- 02_Code: Python code for deep learning network

# Note:
- Spectrogram feature: Log-Mel (Librosa), CQT (Librosa), Gammatone filter (Auditory model)
    + Refer scripts for feature extraction at: https://github.com/phamdanglam/Feature-Extraction-Acoustic-Scene-Classification-DCASE2016
- Framework: Tensorflow 1.X
- Team Ranking in 
    + task 1A - DCASE2019 challenge: 17/36
    
    http://dcase.community/challenge2019/task-acoustic-scene-classification-results-a#teams-ranking    
    + task 1B - DCASE2019 challenge: 5/10    
    
    http://dcase.community/challenge2019/task-acoustic-scene-classification-results-b#teams-ranking    
- Accuracy:
    + Dev set task 1A/B: 76.2/72.9
    + Eva set task 1A/B: 78.8/72.8
