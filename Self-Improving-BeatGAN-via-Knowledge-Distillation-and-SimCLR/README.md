# Self-Improving BeatGAN via Knowledge Distillation and SimCLR

## Overview
 > ECG 데이터에서 비정상적인 리듬을 감지하여 부정맥을 판단할 수 있는 AI기반의 딥러닝 알고리즘인 BeatGAN에 지식 증류 기법을 융합하여 자가 개선을 하는 딥러닝 모델 SI-BeatGAN에, 추가적으로 Contrastive Learning의 학습법을 활용한 SimCRL을 적용해봄으로써 모델의 성능 향상을 추구한다.

#### 지도교수님
- 박경문

#### 팀원
- 2015104236 황채은

---

## 연구 배경
최근 사람들의 주요 관심사 중에 하나로 건강이 자리매김함에 따라 다양한 질병과 그 진단에 대한 중요도가 높아지고 있다. 그 중에서도 심장 질환은 한국인의 10대 사망 원인 중 하나에 속할 정도로 자주 발생할 수 있는 질병이기에, 이를 진단하고 치료하는 것은 병원에 있어서 중대한 일이 되었다. 현재 병원에서는 환자의 심전도 데이터를 측정하고 기록하여 분석하는 일을 전문 의료 인력에게 의존하고 있으며, 이는 많은 인력과 수고로움을 유발한다. 따라서 본 연구에서는 이러한 문제점을 개선하고자 지식 증류를 도입한 BeatGAN의 모델에 추가적으로 대조적 학습법을 도입하여 개선한 부정맥 예측 딥러닝 모델을 제안한다.


## 주요 내용

[BeatGAN]
적대적으로 생성된 시계열을 이용하여 비정상적인 리듬을 감지하는 알고리즘이다. (BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series [1]) BeatGAN은 다음과 같은 장점이 있다. 1) 비지도 방식(Unsupervised) : 레이블 없이 적용이 가능하다. 2) 효율성(Effectiveness) : ECG 데이터에서 거의 0.95 AUC의 정확도를 달성하고 매우 빠른 추론 속도(비트당 2.6ms)를 보인다. 3) 설명 가능성(Explainability): 비정상적인 패턴과 그와 관련된 시간 틱을 정확히 찾아내어 시각화 및 주의 집중을 위한 해석 가능한 출력을 제공한다. 4) 일반성(Generality) : ECG데이터 뿐만 아니라 다변수 모션 캡처 데이터베이스 (CMU Motion Capture Dataset)에서 비정상적인 움직임을 성공적으로 감지한다.

[Knowledge Distillation]
딥러닝에서 지식 증류는 큰 모델(Teacher Network)로부터 증류한 지식을 작은 모델(Student Network)로 전달하는 과정이다. 복잡한 모델은 실제 서비스로 배포할 때 사용자들에게 적합하지 않을 수 있다. 만약 작은 모델이 더 큰 모델만큼의 성능이 나온다면 배포 시 적합하며 컴퓨팅 자원 측면에서도 효율적일 것이다. 지식 증류는 학습 과정에서 큰 네트워크로부터 증류된 지식이 작은 네트워크로 전달하고 그의 성능을 높이는 것에 목적이 있다.
모델이 이미지 클래스를 분류할 때 각 클래스의 확률값이 출력된다. 가장 높은 확률을 보이는 클래스에 따라 예측을 하는 구조로, 교사 네트워크의 분류 결과를 학생 네트워크의 분류 결과와 비교시켜서 학생이 교사를 모방하도록 학습시킨다. 여기에서 예측한 클래스 이외의 확률값에 주목하여 학생 모델이 이러한 정보의 손실 없이 학습할 수 있도록 기존 softmax 함수에 하이퍼파라미터 Τ(Temperature)를 반영한다. 이를 사용하면 낮은 입력 값의 출력을 크게 만들어주고, 큰 입력 값의 출력은 작게 만들어 전체적으로 출력 값을 부드럽게 만들어준다. 이러한 방법으로 지식 증류는 교사와 학생 두 네트워크의 분류 결과를 hard label이 아닌 소프트 레이블(soft label)로 사용하여 학생 네트워크가 학생 네트워크를 모방하여 학습할 때의 이점을 최대화하고 성능을 높인다.


## 세부 사항

[SimCLR]
SimCLR은 Unsupervised Learning인 Contrastive Learning 방식을 사용하는 학습 모델이다. 이 모델은 각 이미지에서 서로 다른 두 Data argumentation을 적용하여, 같은 이미지로부터 나온 결과들은 Positive pair로 정의하고, 서로 다른 이미지로부터 나온 결과들은 Negative pair로 정의하는 형태로 Contrastive Learning 방식을 적용하였다. 이 각 Pair들은 CNN기반의 네트워크(Base Encoder)를 통과하여 visual representation embedding vetor로 변환된다. 이 벡터들은 다시 MLP기반의 네트워크(Projection Head)를 통과하여 변환되고, 변환된 결과들을 이용하여 Contrastive Loss를 계산할 수 있다.
Encoder와 Projection head는 둘 다 batch 단위로 학습을 하며, batch size가 N일 경우 2N개의 sample이 data argumentation을 통해 생성된다. 이러한 방식으로 각 sample별로 1쌍의 Positive pair와 2(N-1)쌍의 Negative pair가 구성된다. SimCLR은 Positive pair간의 유사성은 높이고, Negative pair간의 유사성은 최소화하는 형태로 Loss Function을 구성한다.
일반적으로 Contrastive Learning 방식으로 학습을 할 때, 질이 좋고 충분히 많은 양의 Negative pair를 필요로 하기 때문에 큰 batch size가 필요하다. 따라서 SimCLR은 기본적으로 4096의 batch size(총 8192개의 sample)를 이용하여 학습하며, 128 코어의 cloud TPU를 사용한다. 또한 SimCLR은 큰 크기의 batch size 학습에 용이한 LARS optimizer를 이용하여 분산학습으로 학습하고, Batch normalization을 적용할 때 모든 device에서의 평균/표준편차 값들을 통합하여 적용함으로써, Batch normalization 과정에서 발생하는 정보 손실을 최소화한다


## Results
> 연구결과는 다음과 같다.   
1. 시나리오 1 제안 : SimCLR은 Contrastive Learning이 적용된, 데이터의 label없이 네트워크 모델을 학습할 수 있는 Unsupervised Learning 모델로서, 뛰어난 성능을 보이는 모델이다. SimCLR은 Positive pair와 Negetaive pair를 생성하여 네트워크를 통과시킴으로써 이들을 변환시키고, 변환된 결과를 활용하여 Contrastive Loss를 계산한다.
SimCLR은 여러 번의 연구와 논문을 거쳐 현재 Ver.2까지 발전하였고, 이는 GitHub를 통해 공개적으로 배포되고 있으므로, 활용이 무궁무진하다.
이미 공개적으로 코드가 배포되고 있고, 성능 또한 여러 번의 논문을 통해서 검증된 바 있는 SimCLR을 활용하면, 기존의 SI-BeatGAN의 성능을 향상시킬 수 있을 것이라고 생각한다.
SimCLR은 Contrastive Learning 기법을 활용하여 서로 일치하는 것끼리는 더 가까이 모으고, 서로 다른 것끼리는 더 멀리하게 함으로써 기능하는데, 이를 BeatGAN의 Discriminator에 적용한다면 BeatGAN에서 정상 심전도에서 이상 심전도를 구분해내는 성능이 높아져 AUC를 높일 수 있을 것으로 기대할 수 있다.
따라서, SimCLR의 공개된 코드에서 Contrastive Learning을 수행하는 모델 부분을 기존의 SI-BeatGAN의 Discriminator 부분에 적용하여 결합시킴으로써 모델 성능의 개선을 꾀할 수 있다.

2. 시나리오 2 제안 : 모델의 성능은 손실함수(Loss Function)을 개선하는 것만으로도 유의미하게 올릴 수 있음을 이전 프로젝트를 통해 알 수 있었고, 따라서 이번에도 손실함수를 수정하여 개선함으로써 모델의 성능을 향상시켜 더 높은 AUC를 얻을 수 있을 것이다. 따라서, SI-BeatGAN과 Contrastive Learning의 손실함수들을 적절히 통합하여 모델에 적용할 수 있다면, BeatGAN의 성능은 향상될 것이다. 기존의 SI-BeatGAN은 Contrastive Learning 방식을 적용하고 있지 않은 상태이기 때문에, 이 손실함수의 통합을 통해 Constrastive Learning의 방식도 적용하게 된다면, 주어진 ECG 데이터에서 이상 심전도 데이터를 찾아내는 데에서 더 높은 정확도를 보일 수 있을 것이다.


## Usage
- DataSet (full MIT-BIH dataset)   
  https://www.dropbox.com/sh/b17k2pb83obbrkn/AADzJigiIrottyTOyvAEU1LOa?dl=0  （contain preprocessed data)   
    받은 Dataset은 experiments/ecg/dataset/preprocessed/에 넣는다.   
    
- For ecg full experiement (need to download full dataset)   

    `sh run_ecg.sh`


## Environment Setting
-	OS : Windows 10
-	VGA : Intel(R) HD Graphics 520
-	VGA Driver : 21.20.16.4565
-	Python version : 3.10. (코드 버전 : 3. 7. 12.)


## Conclusion
> 기존 프로젝트의 SI-BeatGAN에 Contrastive Learning 및 SimCLR을 활용하여 성능을 높일수 있는 방안을 연구하였다. 향후에 AUC를 더 향상시킬 수 있도록 개선할 예정이다.


## Reference

[BeatGAN](https://github.com/hi-bingo/BeatGAN)   
[FRSKD](https://github.com/MingiJi/FRSKD)   
[SimCLR](https://github.com/google-research/simclr.git)


## Reports
