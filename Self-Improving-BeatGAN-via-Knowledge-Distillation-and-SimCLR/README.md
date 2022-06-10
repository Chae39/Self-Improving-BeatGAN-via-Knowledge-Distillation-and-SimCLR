# Self-Improving BeatGAN via Knowledge Distillation

## Overview
 > ECG 데이터에서 비정상적인 리듬을 감지하여 부정맥을 판단할 수 있는 AI기반의 딥러닝 알고리즘인 BeatGAN에 지식 증류 기법을 융합하여 자가 개선을 하는 딥러닝 모델 SI-BeatGAN을 제안한다.

#### 지도교수님
- 박경문

#### 팀원
- 2017104019 임에딘
- 2015104236 황채은

---
## 연구 배경
최근 사람들의 건강에 대한 관심사가 늘고 심장 질환에 대한 걱정이 커짐에 따라, 병원에서 부정맥을 진단하고 치료하는 것에 대한 중요도도 커지고 있다. 현재 병원에서는 환자들의 심전도 데이터를 측정 및 기록하고 분석하는 일을 전문의에게 의존하고 있으며, 이는 많은 수고로움을 동반한다. 따라서 본 연구에서는 그러한 문제점을 개선하고자 기존의 비지도 학습 기반 모델을 사용하여 비정상 심전도를 감지하는 BeatGAN에 지식 증류 기법을 도입하여 자가 개선을 하는 부정맥 예측 딥러닝 모델 SI-BeatGAN을 제안한다.

## 주요 내용
우리는 지식 증류 기법을 활용하기 위해 교사 네트워크와 학생 네트워크를 설정했다. 제안하는 모델의 구조는 아래의 그림에 묘사되어 있다. 기존 BeatGAN 모델을 사전에 훈련시키고 이를 교사 네트워크로 삼아서, 데이터에 대해 예측한 결과를 학생 네트워크가 전달받는다. 학생 네트워크는 훈련하는 과정에서 다음 세가지의 손실을 계산 후에 통합하여 최적화 목적을 구성하였으며, 이를 최소화하는 것이 목적이다.

- (1) reconstruction loss : 생성된 x'와 입력데이터 x와의 복원 오차
- (2) adversarial loss : 특징 일치 손실(feature matching loss, D의 은닉층 활성화 벡터를 이용하여 생성된 x'와 x의 특징 차이 고려)
- (3) knowledge distillation loss : 학생이 예측한 결과와 교사로부터 전달받은 결과와의 손실을 통합하여 업데이트를 진행한다.

![network_structure](https://user-images.githubusercontent.com/30232133/139444754-0601fb02-8d21-4acc-a2a8-45cd8207aa8e.jpg)   
[그림] SI-BeatGAN 네트워크 구조


## 세부 사항
#### [네트워크 구조]
1. 오토인코더는 인코더G_E(Generator Encoder)와 디코더G_D(Generator Decoder)로 구성되어 있으며, 입력 데이터 x가 인코더로 들어오면 은닉 벡터 z로 압축되고 G_D (z)는 x'를 생성한다. 여기에 적대적인 학습 방식인 GAN 기술을 추가하여 Discriminator(D)가 정규화의 역할을 수행한다.
2.  디코더 네트워크인 G_D의  구조는 DCGAN의 생성자 구조와 비슷하다. DCGAN은 기존 GAN에 존재했던 완전 연결된 구조의 대부분을 CNN구조로 대체한 것이다. G_D는 그림 2와 같이 1D transposed 컨볼루션 레이어들과 Batch-norm, Leaky ReLU로 구성되어 있다. G_E 는 G_D와 거의 유사하며 반대 방향으로 되어있다.    
![GD](https://user-images.githubusercontent.com/30232133/139444611-d03b7687-947a-46f4-bc41-8e1bfe5fee6d.jpg)   
[그림] 디코더 네트워크 G_D구조



## Results
연구결과는 다음과 같다.   
<br>
|Model|AUC |AP |
|---|---|---|
|BeatGAN|0.9458|0.9108|
|SI-BeatGAN|0.9474|0.9148|

[표] 기존의 BeatGAN 모델로 사전에 훈련을 시킨 후에 테스트한 결과와, 사전 훈련된 교사 네트워크를 이용하여 지식증류 기법으로 학생 네트워크를 훈련시킨 후 테스트한 결과를 표로 나타낸 것이다. 기존의 모델보다 자가 개선을 수행하는 SI-BeatGAN의 정확도가 향상되었음을 알 수 있다.   


## Usage
- DataSet (full MIT-BIH dataset)   
  https://www.dropbox.com/sh/b17k2pb83obbrkn/AADzJigiIrottyTOyvAEU1LOa?dl=0  （contain preprocessed data)   
    받은 Dataset은 experiments/ecg/dataset/preprocessed/에 넣는다.   
    
- For ecg full experiement (need to download full dataset)   

    `sh run_ecg.sh`

## Environment Setting
- python 3.7.6
- PyTorch 1.7.1
- cuda 11.0
- cudnn 8.0.5
- GPU : GTX1660super
- CPU : i5-10400f

## Conclusion
> 기존의 BeatGAN에서 교사 네트워크와 학생 네트워크 구성을 하고 지식 증류 기법을 융합하여 자가 개선을 하는 SI-BeatGAN을 소개하였다. 학생 네트워크는 훈련하는 과정에서 학습된 교사 네트워크의 지식을 전달받아 자가 개선을 수행하였으며 그 결과 기존 모델보다 더 높은 정확도를 보여주었다. 향후에 AUC를 더 향상시킬 수 있도록 개선할 예정이다.

## Reference

[BeatGAN](https://github.com/hi-bingo/BeatGAN)   
[FRSKD](https://github.com/MingiJi/FRSKD)   

## Reports
