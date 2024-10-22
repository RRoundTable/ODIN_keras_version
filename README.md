# ODIN_keras_version
Out-of-distribution을 검출하는 방법론 중 하나인 odin(pytorch)을 keras화 하였습니다.


# ODIN

out-of-distribution을 검출하는 Odin의 장점은 pretrained된 model을 그대로 사용할 수 있다는 점입니다.

기존의 pretrained model에 추가적으로 T scaling과 perturbation을 적용하여,

in-distribution과 out-of-distribution을 구분할 수 있습니다.

reference : https://github.com/facebookresearch/odin


# result

### 아래 그래프를 보면, ROC커브가 개선되었음을 알 수 있다.
  (Temperature=1000, Magnitude=0.0014)
  
![graph](https://user-images.githubusercontent.com/27891090/54009777-7f293500-41af-11e9-8daa-bcf22e7d16b0.png)

- var 는 activation map vaule의 분산을 기준으로한 검출방법론입니다.
- max 는 activation map value의 max를 기준으로한 검출방법론입니다.
### 실험 상세 내용-1

- In-distribution dataset: CIFAR-10
  
- Out-of-distribution dataset: Tiny-ImageNet (crop)
 
  | Tables   	|      Baseline      	|  ODIN 	|
  |----------	|:-------------:	|------:	|
  | FPR at TPR 95%:  	| 14.9% 	| 2.1%  	|
  | Detection error: 	|    8.3%   	|  3.8% 	|
  | AUROC: 	| 97.4% 	|    98.6%	|
  | AUPR In:|         97.8%       	|   99.0%    	|
  | AUPR Out:|        97.1%        	|    99.7%   	|
  
  
### 실험 상세 내용-2

- In-distribution dataset: CIFAR-10  

- Out-of-distribution dataset: CIFAR-100 
 
  | Tables   	|      Baseline      	|  ODIN 	|
  |----------	|:-------------:	|------:	|
  | FPR at TPR 95%:  	| 24.3% % 	|  6.7%	|
  | Detection error: 	|   12.0%   	|  6.9% 	|
  | AUROC: 	| 94.6% 	|    94.8%	|
  | AUPR In:|        95.0%        	|    95.3%    	|
  | AUPR Out:|        94.9%        	|    99.3%   	|
  
  
  # Activation Map
  
  ### in-distribution
  
  ![image](./activation_map/activation_map_70_in.png)
  
  ### out-of-distribution
  
  ![image](./activation_map/activation_map_3_out.png)
  
 
  out-of-distribution의 activation map은 in-distribution과 큰 차이를 보이고 있다.
  
  상대적으로 분산된 activation map을 가지는데 이는 classification을 할 때, 중요하게 생각하는 포인트를 못 잡아내는 것으로 해석할 수 있다.
  
 
  ### reference : http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf
  
  
