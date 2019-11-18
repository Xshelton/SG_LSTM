# SG_LSTM_Frame

## 1.project information 
1. @ARTICLE{SG-LSTM-FRAME,
2. author = {Weidun Xie(shelton xie),Jiawei Luo,Chu Pan,Ying Liu},  
3. title = {SG-LSTM-FRAME: A Frame Using Sequence and Geometry information via LSTM to predict gene-miRNA associations},
4. year = {2019},  
5. journal = {},}  

## 2.Environment: 
- Hardware: Core: Intel i7-7700HQ ;  Graphic Card：GTX1060 6G;  RAM:32 GB ddr4 2666
- CUDA Version 10.0.130
- CuDNN version 7.4.2
- OS: Win 10 64bit
- Python version: IDLE (Python 3.7 64-bit)#yep I wrote directly in IDLE :)
## 3.Needed python package: 
- PyQt5                5.10.1(for UI)
- Pandas               0.24.1(for file read)
- Sklearn              0.0(for comparison)
- tensorflow-estimator 1.13.0(for LSTM)
- tensorflow-gpu       1.13.1(for LSTM)
- numpy                1.16.2(for npy read and other functions)
- matplotlib           3.0.3(for roc curve plot)


## 4.How to run the file:
This is the code of SG_LSTM_core, the only difference between SG_LSTM_core and SG_LSTM_WHOLE is the difference between dataset and the way we store the unknown pairs' score.
### 1.Fast train:
-  print('In this test, The dataset is randomly separated into train and test set, you don’t have to generate other file')
-  what you need is the dataset downloaded or generated use **SG_LSTM_Dataset UI**
-  then change **RNN(LSTM) - fast validate-train.py**  **your_dataset_file_name='AXB_383_gene_default.csv'** into 
-  your_dataset_file_name='your file generated.csv'
-  open the **RNN(LSTM) - fast validate-train.py** in IDLE and Press **F5**
-  It will automatically begin to train ,the model will be saved in the same folder
#### if you don't want to generate new dataset AXB_383_gene_default.csv can be downloaded either 
AXB_383_gene_default.csv 159.1MB 
link：https://pan.baidu.com/s/1Yy8RNfinV2B0G9l_Gb8-Ag  password：11v1 
### 1.1 Fast test:
- In order to fast test: first put **RNN(LSTM)-fast-validate-test.py** and **AXB_383_gene_default.csv** and **models** in the same folder.
-  Then change:
-  your_dataset_file_name='AXB_383_gene_default.csv' into your_dataset_file_name='your file generated.csv' **if you decide to generate your own dataset.** otherwise ,skip this step
-  Then change:
-  your_file_path="C://Users/shelton/Desktop/fast validate/models"  into your file root , otherwise it cannot find the model.
-  After the model successfully loaded, test will automatically run. 
-  it will produced the y_score and y_label for your test set, named LSTM_y_score and LSTM_y_label ,and also after running the **RNN(LSTM)-fast-validate-test.py**,a ROC -curve would appear to indicate the performance of the model.


### 2.Five fold cross validation Experiments:
- 0 SG-LSTM(for test the model).py can help to build the model
- 0 SG-LSTM(for roc curve generation).py can use the model to do the test job, (generation of ROC curve)
also the scores!
- 1.put SG-LSTM(for test the model).py and X_train5.csv and X_test5.csv in the same folder
- 2.run SG-LSTM(for test the model).py (after a while of running, a folder named model5 would be generated)
- 3.put SG-LSTM(for roc curve generation).py and X_train5.csv and X_test5.csv in the same folder
- 4.You have to change the file root so that the roc curve could be generated successfully:
- From module_file = tf.train.latest_checkpoint("I://DNN模型重新测试/LSTM测试/测试24 SG相乘128 五折交叉验证/LSTM for test/models{}".format(gg))
- To module_file = tf.train.latest_checkpoint("C://Users/shelton/Desktop/SG-LSTM_master/models{}".format(gg))
- here 'gg' means the iteration of fold. gg=5 means the last fold of SG-LSTM-core
- please change it into your file root, otherwise it can not load the model.
 
 ### X_train5 and X_test5 can be download :
- X_train5.csv
 links：https://pan.baidu.com/s/1ZvlfgQ8IjEo61extIU6WvQ 
 password：5o11 
- X_test5.csv
links：https://pan.baidu.com/s/1X7OmbCQq8OeudzWmkZydqQ 
password：ld7z 
### And can also be generated using 
 - **5_fold_cross_divided.py** and  **'AXB_383_gene_default.csv'**
 ## 5.Some Material for our framework(like dataset, and trained embedding)
 - **5_fold_cross_divided.py** can divide **'AXB_383_gene_default.csv'**(or other dataset) into five fold of train and test parts.
 - **all_unkown_sample_generation.py** can generate all negative samples for prediction. Here in SG-LSTM-core, we generate all unknown samples by generating file. But in the SG-LSTM-WHOLE, because there are more than 14,000,000 pairs, we generate the score by producing a matrix to record the scores.
 
- S_gene2vec_320_34567.csv S_mrna2vec_384_34567.csv is the embedding generated from sequence information of 320 gene and 384miRNA
- G_gene2vec_328_gene.csv G_mirna2vec_384_mirna.csv is the embedding generated from geometric information.

- S_gene2vec_320_34567.csv+G_gene2vec_328_gene.csv can generate SG128_317_gene.csv.
- S_mrna2vec_384_34567 +G_mirna2vec_384_mirna.csv can generate SG128_383_miRNA.csv

- SG128_317_gene.csv is the embedding merged from Geometric information embedding and Sequence information embedding of gene.
- SG128_383_miRNA.csv is the embedding merged from Geometric information embedding and Sequence information embedding of miRNA.

- Using SG128_317_gene.csv and SG128_383_miRNA.csv and a file from mitarbase, you can generate positive samples;
After calculate all the pairs' Euclidean distance and cosine-similarity, you can earn a average distance. Using the positive samples and distance, you can generate the dataset, like 'AXB_383_gene_default.csv'

- 'AXB_383_gene_default.csv' is the training Dataset of SG-LSTM-core.


<p>
  
# SG_LSTM_Dataset UI 
I designed an UI for dataset construction. It contains all the content about how to use SG128_317_gene.csv and SG128_383_miRNA.csv to generate file like  'AXB_383_gene_default.csv'
- SG_LSTM_Dataset UI  https://github.com/Xshelton/SG_LSTM_DatasetUI

<p>
 
 # SG-LSTM_core_result
 #If your only want the results of my research:
 - SG-LSTM_core_result: https://github.com/Xshelton/SG_LSTM_core_result
 
 <p>
 
  # SG-LSTM_WHOLE_result
  In the WHOLE part,more than 14,000,000oairs of miRNA and Gene were predicted.
 - SG-LSTM_WHOLE_result https://github.com/Xshelton/SG-LSTM_WHOLE_result
