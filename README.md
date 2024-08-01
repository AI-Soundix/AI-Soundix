# AI-Soundix
__다양한 소리데이터를 대상으로 하는 웹 기반의 AI 통합 솔루션 및 관제 프레임워크__

![AI-Soundix](https://github.com/user-attachments/assets/e0d35915-f615-4a24-a8c2-106480a44c5b)

* 다양한 소리데이터의 특성을 고려한 효율적 처리 및 분석 기술
* 소리데이터 전주기(수집-저장-처리-분석)의 통합적인 AI 플랫폼
* 응용 맞춤형 AI 모델을 제공함으로써 회귀/이상탐지/분류 등 특정 목적에 맞는 정밀한 분석과 예측 지원

<br/>

## 1. 층간소음 충격원 분류
### Setup
* `Python 3.10.14`
* `Numpy 1.26.4`
* `Pytorch 2.4.0`
* `Torchvision 0.19.0`

<br/>

빠른 Test를 위한 Trained [Resnet50](https://drive.google.com/file/d/1CE1GtbhxL3hSRNkJF5IsuhtWd2em2CIC/view?usp=drive_link)

<br/>

### Dataset
.wav 파일을 전처리를 통해 이미지로 변환한 것을 모델의 입력으로 사용

<br/>

__Dataset Format__

```bash
classification
├── dataset
│   ├── train
│   ├── test
│   └── valid
├── test.py
├── util.py
└── dataset.py
```

인하대학교 지능형 임베디드 소프트웨어 연구실에서 수집한 층간소음 [Dataset](https://drive.google.com/file/d/1_N_IJ5lJifwEH9UlA-oHWjd15oZ0pgQt/view?usp=sharing) 또는 위 Dataset Format에 맞춘 Custom Dataset 사용

<br/>

### Running the Application
1. Trainning
   ```
   python test.py
   ```
2. Test
   ```
   python test.py --mode "test"
   ```
