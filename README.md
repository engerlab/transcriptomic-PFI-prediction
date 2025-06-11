# Structure

All coding files are contained in the src/ folder. Different labeling strategies are provided in the labels/ folder. The labels used in this publication are contained in PFI.csv.

The data used can be downloaded from the GDC (https://portal.gdc.cancer.gov/) following the instructions laid out in the paper. 

To work without making any changes to the code the data should be stored in the following structure

Genomic_Outcome_Prediction_Public/
├─ data/
    ├─ Clinical/
    |    ├─ Clinicaldata/
    |    |    ├─ ** downloaded clinical data following instruction
    ├─ mRNA/
    |    ├─ downloads
    |    |    ├─ primary_tumor
    |    |    |    ├─ ** downloaded mRNA data following instruction
    ├─ miRNA/
    |    ├─ downloads
    |    |       ├─ primary_tumor
    |    |       |     ├─ ** downloaded mRNA data following instruction
├─ labels/
├─ src/


Otherwise the regex expressions in *_pipe.py files will need to be changed.

# Files 

## clinical_pipe.py

This file contains the logic to load and train RSF on clinical only data. Including the data mapping to integer values, which datatypes were included, normalization, imputation, and labeling of patients. Contributes data to Figures 5, 8, and 9. After training model scores are printed to the terminal and were recorded manually. Stratification and feature importance results are saved for further analysis.

## cohort_size.py

This file contains a simple script to calculate the data used in Fig.1 

## extra_analysis.ipynb 

An additional to generate figures after the models have been trained and evaluated. Generates Figure 9 which was then annotated with additional information

## miRNA_pipe.py

This file contains the logic to load and train RSF on the miRNA data only. Includes the normalization, feature selection algorithms and their potential hyperparameters. After training model scores are printed to the terminal and were recorded manually. Stratification and feature importance results are saved for further analysis.

## mRNA_pipe.py

This file contains the logic to load and train RSF on the mRNA data only. Includes the normalization, feature selection algorithms and their potential hyperparameters. After training model scores are printed to the terminal and were recorded manually. Stratification and feature importance results are saved for further analysis.

## multi_pipe.py

This file contains the logic to load and train multimodal RSF models. Early fusion and Late fusion strategies are included as well as methods to try any combination of data types. Loading logic is imported from the other _pipe files. After training model scores are printed to the terminal and were recorded manually. Stratification and feature importance results are saved for further analysis.

## NN_training.ipynb

This file contains the equivalent of all _pipe files but uses DeepSurv instead of a RSF

## NN.py

Some basic functions used only in NN_training.ipynb

## train.py

This file contains functions concerned with testing and evaluating different models.

## util.py

This file contains different feature selection algorithms and methods for processing data that are generally useful.