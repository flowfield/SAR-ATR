# SAR-ATR
인공지능 기반 SAR ATR 기술 개발을 위해
1. 기존 패턴 매칭 기반의 low level classifier를 구현하여 비교군을 생성
2. 이종 데이터를 이용하여 transfer learning
3. 학습을 위한 SAR ATR 데이터 확보 및 획득
--------------------------------------------------------------------------------------------------------
패턴매칭 기반 기술 구현

![image](https://user-images.githubusercontent.com/105264422/167564211-96862356-b2e1-4bf7-ac61-0e4914000fbd.png)
FOA를 이용한 ATR 기술 구현

![image](https://user-images.githubusercontent.com/105264422/167564296-6c2cabf5-b255-4ed9-acff-29629c525a48.png)
filter를 이용하여 edge를 보존하면서 전처리
Constant False Alarm Rate(CFAR) 알고리즘을 통하여 빠르게 후보군을 추림
ORB 혹은 SIFT 알고리즘을 통해 이미지 내의 keypoint와 descriptor를 뽑아냄
FLANN 매치로 특징점 비교 후 매칭

--------------------------------------------------------------------------------------------------------
데이터 획득
conditional GAN을 통해 MSTAR 데이터와 비슷한 이미지 생성 후 학습 데이터로 활용

--------------------------------------------------------------------------------------------------------
인공지능기반 ATR 기술
1. EO/IR 영상을 활용한 모델 선행학습 후 classifier를 따로 학습하는 transfer learning 방법 이용
2. Data augmentation을 이용하여 더 많은 학습 데이터 이용 및 over fitting 방지

--------------------------------------------------------------------------------------------------------
진행 사항
![image](https://user-images.githubusercontent.com/105264422/167565247-8387f512-9611-42a7-84d9-03ed948bd074.png)
브라질 산토스 지역 SAR 이미지 획득

![image](https://user-images.githubusercontent.com/105264422/167565303-e1736b5c-4e8b-49e2-8f5d-4f14b8fc26db.png)
![image](https://user-images.githubusercontent.com/105264422/167565322-e72c2179-5f6c-4c9d-8a48-2baddca97805.png)
ENL과 PSNR 모두 킬수록 노이즈가 적음을 의미
Guided filter가 가장 좋은 결과를 보임

![image](https://user-images.githubusercontent.com/105264422/167565482-a1ee04c8-7282-4243-b698-a94c15846ac2.png)
CA-CFAR 알고리즘으로 feature 골라내어 데이터 량을 감소시킴

![image](https://user-images.githubusercontent.com/105264422/167565600-e41f9db2-63eb-44d4-9491-dd9d6e3232c2.png)
conditional GAN의 한 종류인 pix2pix 구현, 이후에 MSTAR 데이터에 활용 예정
