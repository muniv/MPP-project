# How to run
프로젝트 실행 환경:

- **GPU:** NVIDIA A100
- **CUDA 버전:** 12.1
- **Python 버전:** 3.8
- **torch 버전:** 2.0.1+cu117

### Pretraining dataset
main_PCQM4.py   
--train_subset: Whether to use only a fraction of train data  
--gnn: select the gnn type  

### Finetuning dataset
1. **BBBP 원본 프로젝트 Finetuning:**
   - `main_bbbp.py`: BBBP 프로젝트에서 기존에 사용한 코드.
2. **MPP 프로젝트 Finetuning:**
   - `main_bbbp_mpp.py`: 이번 MPP 프로젝트에서 수정한 코드.

train/val/test dataset is already in the folder.