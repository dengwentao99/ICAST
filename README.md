# Paper summary

## Task

The task of answer selection is to select an answer from candidate answers to satisfy the users' information needs. 

## Model
<img src=./img/Method.png width=50% />

Architecture of our proposed intent-calibrated self-training (ICAST) framework. The dashed and solid line represent the workflow of teacher model and student model, respectively. The blue and green solid line represent intent-aware and context-aware workflow, respectively. The intent-calibrated pseudo labeling module estimates intent confidence gain to select samples with high-quality intent labels, and calibrates the answer labels by incorporating selected intent labels as an extra input for answer selection.

# Running

## Requirements

```
Python == 3.9.7
torch == 1.11.0
apex == 0.1
scipy == 1.8.0
transformers == 4.17.0
accelerate == 0.9.0
```



## Datasets

1. We use [MSDIALOG](https://share.weiyun.com/JezuHlHU) and [MANTIS](https://share.weiyun.com/1ezb9Srg) datasets for training and testing. 
2. The datasets consist of four subfolders: teacher/MSDIALOG, student/MSDIALOG, teacher/MANTIS and student/MANTIS. 
3. After downloading the datasets, place them into /datasets in teacher/MS-dialog, student/MS-dialog, teacher/Multi-domain-IS and student/Multi-domain-IS, respectively. 
4. Please download the pre-trained model [BERT-base-uncased](https://huggingface.co/bert-base-uncased) into prev_trained_model/bert-base-uncased. 

## Training and testing

#### Training the teacher model

```
## MSDIALOG dataset

cd teacher/MS-dialog
# Training
bash scripts/run_training.sh
# Testing
bash scripts/run_testing.sh
```

```
## MANTIS dataset

cd teacher/Multi-domain-IS
# Training
bash scripts/run_training.sh
# Testing
bash scripts/run_testing.sh
```



#### Training the student model

```
## MSDIALOG dataset

cd student/MS-dialog
# Training
bash scripts/run_training.sh
# Testing
bash scripts/run_testing.sh
```

```
## MANTIS dataset

cd student/Multi-domain-IS
# Training
bash scripts/run_training.sh
# Testing
bash scripts/run_testing.sh
```

To use our codes, train the teacher model first and select best teacher model with development sets. After that, put the checkpoint of teacher model into the folder of pre-trained model. Finally, train the student model with labeled and unlabeled datasets and select the best student model with development sets. After training, select threshold of probabilities for each experimental setting. 
