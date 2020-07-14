# Human-Driver-Behavior-Classification-from-Partial-Trajectory-Observation
This is my master thesis work for my MS in Mechanical Engineering at Carnegie Mellon University, advised by Prof. Changliu Liu at the Intelligent Control Lab of The Robotics Institute.

The write-up .pdf file which explains the proposed method and the pipeline is in the repo.
![alt text](https://github.com/angmavrogiannis/Human-Driver-Behavior-Classification-from-Partial-Trajectory-Observation/blob/master/Pictures/pipeline.JPG)

Here is a 7-minute video introducing the idea: https://youtu.be/ldNasL2I--A

**Requirements:**
- numpy
- pandas
- pickle
- sklearn
- matplotlib
- seaborn
- torch

**Instructions**

Run the files with the following order:

- preprocessing.py
- more_preprocessing.py
- clustering.py

- lstm_extract.py
- train_lstm.py

The first 3 scripts implement the behavior classification.
The last 2 scripts apply the behavioral analysis to the trajectory prediction problem.
