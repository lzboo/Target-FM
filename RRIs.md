# miRNA-mRNA interactions

> lzboo 2022/9/16

## I Dataset

* PAR-CLIP and CLASH experiment datasets

### 1.1 Train

* **site-level**
  
  - **positive**: 33,142
  
  - **negative**: 32,284
  
  <img src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/09/01-10-33-56-1.png" title="" alt="1.png" data-align="center">

### 1.2 Test

* 10 test sets

* **gene-level**
  
  - **positive**: 151,956
  
  - **negative**: 548 (randomly sampled 548 positive pairs for ten times to balance the number of positive and negative pairs)
  
  <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/09/01-10-35-01-4.png" alt="4.png" data-align="center" width="381">

## II Experiment

### Result

|             | DeepTarget(bs=32) | Bert + fc (bs=32) | Bert + 1d-cnn(bs=32) | Bert + fc (bs=128) | Bert + 1d-cnn(bs=128) | Bert + 1d-cnn(bs=128 + lr_sch) |
|:-----------:|:-----------------:|:-----------------:|:--------------------:|:------------------:|:---------------------:| ------------------------------ |
| set1        | 0.7729            | 0.8208            | 0.7954               | 0.7811             | 0.8094                | 0.8237                         |
| set2        | 0.7934            | 0.8338            | 0.8323               | 0.8349             | 0.8332                | 0.8393                         |
| set3        | 0.7948            | 0.8364            | 0.8310               | 0.8141             | 0.8380                | 0.8292                         |
| set4        | 0.7690            | 0.8376            | 0.8232               | 0.8287             | 0.8382                | 0.8293                         |
| set5        | 0.7707            | 0.8489            | 0.8451               | 0.8387             | 0.8553                | 0.8371                         |
| set6        | 0.7758            | 0.7795            | 0.7925               | 0.7758             | 0.8183                | 0.7918                         |
| set7        | 0.7787            | 0.7883            | 0.7895               | 0.7917             | 0.8100                | 0.7993                         |
| set8        | 0.7822            | 0.8218            | 0.8169               | 0.8057             | 0.8324                | 0.8247                         |
| set9        | 0.7753            | 0.8171            | 0.8233               | 0.8157             | 0.8396                | 0.8293                         |
| set10       | 0.7907            | 0.8308            | 0.8397               | 0.8329             | 0.8469                | 0.8345                         |
| **average** | **0.7804**        | **0.8215**        | **0.8189**           | **0.8119**         | **0.8321**            | **0.8238**                     |

#### Options

* `downstream sturcture`
  
  - [x] TargetFM1: Bert + fc<img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/09/16-10-39-37-fc.jpg" alt="fcjpg" data-align="center" width="461">
  
  - [x] TargetFM2: Bert + 1dCNN<img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/09/16-10-50-35-cnn.jpg" alt="cnnjpg" width="458" data-align="center">
  - [x] TargetFM2: Bert + BiLSTM

* `batchsize` = [32, 128]

* `epoch` =10
- `level` = 'gene'

- `seed_match` = 'offset-9-mer-m7'

- `optimizer` = `optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)`
* `lr_scheduler` 
  
  - [x] CosineAnnealingLR£º
    
    ```python
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    ```
  
  - [ ] StepLR:
    
    ```python
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    ```
  
  - [ ] MultiStepLR:
    
    ```python
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,80],gamma = 0.9)
    ```
  
  - [ ] ExponentialLR:
    
    ```python
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    ```
  
  - [ ] ReduceLROnPlateau:
    
    ```python
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    ```

* `max_length`



#### Appendix

### Result

1. **TargetFM1: Bert + fc** 
   
   * `lr` = 1e-3
   - [x]  `batch_size` = 32
     
     <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/09/16-15-37-02-test_fc_32_1e-3.jpg" alt="test_fc_32_1e-3.jpg" width="312" data-align="center">
   
   - [ ]  `batch_size` = 128
     
     <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/09/16-17-02-40-test_fc_128_1e-3.jpg" alt="test_fc_128_1e-3.jpg" width="308" data-align="center">
     
     

2. Target2: **TargetFM1: Bert + 1dcnn**
   
   * `lr` = 1e-4
   - [x] `batch_size` = 32
     
     <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/09/16-14-30-35-test_cnn_32.jpg" alt="testcnn32jpg" width="346" data-align="center">
   
   - [x] batch_size = 128
     
     <img title="" src="https://raw.githubusercontent.com/lzboo/ImgStg/main/2022/09/16-14-57-50-test_cnn_128.jpg" alt="test_cnn_128.jpg" width="338" data-align="center">

3. TargetFM3: BiLSTM
