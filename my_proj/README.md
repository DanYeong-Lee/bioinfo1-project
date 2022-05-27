# LIN28A binding motif prediction

## 개요
- 최근 일정 길이 DNA sequence를 input으로 받는 deep learning task를 하나 진행중이었음.
- 그래서, 유사하게 RNA sequence data를 input으로 받는 motif prediction task를 주제로 선정하였음.


## 계획

### Task
- Input: 20nt RNA sequences
- Output: LIN28A binding motif or not? (binary classification)

### Data
- Positive samples: CLIP-seq에서 얻은 LIN28A-bound sequences
- Negative samples: ???

- One-hot encoding 예정
  
### Models
- Baseline: Logistic regression, SVM, RF, simple MLP 등등 예정
- My model: 1d-CNN or RNN/Transformer 기반 딥러닝 모델 구축 예정
  - 아마 간단히 "Input -> CNN/RNN -> flatten으로 vectorize -> MLP -> prediction" 의 구조로 구현할 듯
  
### Evaluation
- Metrics: Accuracy, AUROC, ... 
- Sample 수를 구해봐야 알겠지만, 데이터 양에 따라 그냥 test data를 떼어놓고 시작할지, 혹은 cross-validation scheme을 따를지 정할 예정 
  
### Implementation
- Scikit-learn
- PyTorch & PyTorch Lightning
  
  
## 해결해야 할 문제

- Negative samples를 어떻게 정의하고 얻을 것인가?

  - LIN28A binding motif가 아닌 sequences여야 함.
  - Ideas
    1. Randomly generated sequences?
    2. CLIP-seq의 결과 LIN28A의 binding region이 아닌 다른 random region에서 20nt sequences randomly 가져오기?
  
  - 어떤 방법을 쓰던, Positive samples와 개수를 맞추고 싶음
