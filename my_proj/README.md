# LIN28A binding motif prediction

## 개요
- 최근 일정 길이 DNA sequence를 input으로 받는 deep learning task를 하나 진행중이었음.
- 그래서, 유사하게 RNA sequence data를 input으로 받는 motif prediction task를 주제로 선정하였음.


## 계획 및 진행 상황

### Task
- Input: 20nt RNA sequences
- Output: LIN28A binding motif or not? (binary classification)

### Data
- Positive samples: CLIP-seq에서 얻은 LIN28A-bound sequences
  - CLIP-35L33G.bam file
  - 논문에서와 같이 Shannon entropy > 0.8, read counts > 50인 position을 filtering한 이후,
  - 해당 position의 flanking sequence (20nt)를 얻을 예정
  - 총 14,901개 position을 얻었음 -> GRCm39 genome에서 해당 position들 주변 20nt sequence들을 얻어올 예정
    - Strand 주의해서 가져와야 함.

- Negative samples
  - 확실한 negative sample을 만들기 위해, transcription은 되지만 LIN28A와는 확실히 binding하지 않는 부분에서 가져와야 할 듯
  - CLIP-35L33G.bam file에 한번도 나타나지 않는 genomic region 중 gene region을 쭉 concat한 다음에 randomly 20nt sampling?
    - 혹시 사실은 LIN28A가 가서 붙지만 CLIP-seq의 결과로 알아내지 못한 sequence도 있을 수 있으므로, randomly sampling한 다음에 혹시나 positive sample에 똑같은 sequence가 있는지 한번 확인은 해야 할 듯?

- One-hot encoding 예정
  - CNN or RNN 기반 모델 말고, 다른 모델들에게는 k-mer frequency로 vectorize해서 넣어주는 게 맞을 듯.
  
### Models
- Baseline: Logistic regression, SVM, RF, simple MLP 등등 예정
- My model: 1d-CNN or RNN/Transformer 기반 딥러닝 모델 구축 예정
  - 아마 간단히 "Input -> CNN/RNN -> flatten으로 vectorize -> MLP -> prediction" 의 구조로 구현할 듯
  - Sample이 그닥 많지 않으므로 그리 깊지 않게 만들어봐야겠다.
  
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
    2. CLIP-seq의 결과 LIN28A의 binding region이 아닌 다른 random region에서 20nt sequences randomly 가져오기? (이걸로 결정)
  
  - 어떤 방법을 쓰던, Positive samples와 개수를 맞추고 싶음
  - 아무래도 randomly generated sequence보다는, 그래도 realistic한 sequence를 negative sample로 사용하고 싶어서, 2번으로 해야 할 듯.

- Positive strand & Negative strand에 mapping된 것을 구별해야 할 듯??
  - 보니까 mpileup 결과에서 positive strand에 mapping되면 대문자, negative strand에 mapping되면 소문자로 표현된다고 함.
  - 이걸 잘 보고 맞는 strand에서 가져와야겠다
