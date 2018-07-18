# Weight Initializaiton

* Weight 초기화를 어떻게 하느냐에 따라 Neural Network의 성능이 달라짐.
* Weight 초기화를 제대로 하지 않으면, 모든 activation이 zero가 됨[4].



###1. RBM

* 2006년에 Hiton 교수는 [1]에서 RBM을 통해 Weight 초기화 하는 방법을 제시함. 
* RBM이 성능 측면에서 괜찮았지만, Pretraining이 필요해 많은 시간이 소요된다는 단점이 있었음.



### 2. Xavier Initialization 

* 2010년에 Xavier 교수는 [2]에서 RBM의 단점을 보완한, 일명 Xavier initialization(혹은 Glorot initialization라고도 불림.)을 선보임. 
  * 매우 간단한 initialization임에도 불구하고 잘 working 했음.
* Xavier initialization은 Weight의 분산을 정하는 초기화 과정.
  * 분산이 크면 클 수록 더 많은 Space를 탐색하게 되므로 성능이 좋아질 가능성이 높지만,
  * 분산이 너무 크면, Gradient Exploding 현상 때문에 항상 Saturation이 일어날 수 밖에 없음. 
  * 그래서 분산을 잘 정하는게 중요함.
  * Xavier initialization은 activation function이 linear 하다고 가정할 때 최적의 분산 값을 가지도록 제안된 초기화 과정임.(실제로는 activation이 linear하지 않지만..)
    * 그래서 non-linear activation function에 대해 잘 되는 지에 대해서는 실험적으로 증명했다.
* Xavier initialization은 weight matrix의 각 element의 분산을 다음과 같이 초기화 한다.
  * `w`는 특정 layer의 weight matrix이고,
  * `fan_in` 는 해당 layer의 input layer unit 개수,
  * `fan_out` 는 해당 layer의 output layer unit 개수

```latex
Var[w] = 2 / (fan_in + fan_out)
```

* (TMI) 위 식이 저렇게 되는 이유는 [2] 논문에서 10번 11번 12번 식을 보면 알 수 있음.
  * 10번 조건과 11번 조건을 모두 만족해야 하는데, `fan_in != fan_out` 이면, 해당 두 조건을 모두 만족하는 해를 찾을 수 없으므로,
  * 12번 식과 같이 평균낸 식을 쓰는 것이 합리적이란 것이 논문의 설명.
* 스탠포드 대학교의 CS231n 코스에서도 그렇고 몇몇 자료에서는 위 식을 다음과 같이 표기하기도 함.
  * input layer의 unit 수와 output layer의 unit 수가 같다고 가정한 것임.
    * `fan_in == fan_out`
  * 엄밀하게 말하면 틀린 식일 수 있지만, Xavier initialization자체가 activation function이 linear 하다는 가정 아래 탄생한 식이기 때문에 저정도의 다름은 그냥 인정하고 넘어가는 듯 함.
  * 실제 테스트에서도 별 문제 없고, 간단하게 표기하기 위해 아래와 같이 표기(더 나아가 구현까지)하는 듯.

```latex
Var[w] = 1 / fan_in
```

* [Xavier initialization의 Tensorflow 구현체](https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/contrib/layers/python/layers/initializers.py)를 보면, uniform하게 initialize 할 것인지 정하게 되어있음. 
  - uniform하게 초기화 하지 않는 경우 normal distribution으로 초기화 함. 
  - **Xavier initialization의 경우 분산을 맞추는 데 초점이 맞춰져 있으므로, uniform인지 normal인지 중요치 않음.**
  - (이건 내 생각) 하지만, 구현체에서 잘 보면 normal distribution을 truncated normal distribution 함수를 이용해 구현했음. 
    - truncated normal을 쓰지 않으면, 너무 극값이 나와 계산이 제대로 되지 않는 경우가 발생할 수 있음.
    - 하지만, truncated normal을 쓰면 너무 극한의 값들은 제거되기 때문에 분산이 변경될 여지가 있음. 
    - 그래서, uniform distribution을 쓰는 편이 더 안전할 듯. 
    - 비슷한 이유에서 tensorflow 구현체에서도 uniform을 default로 설정하지 않았을까?
  - 아래 코드에서 uniform distribution의 식이 저렇게 되는 이유는 확률와 통계 책 uniform distribution의 분산 계산과정을 보면 이해가 될 것임.

```python
std = np.sqrt(2 / (fan_in + fan_out))
w = np.random.normal(0, std)	# normal distribution 식
w = np.random.uniform(-std * np.sqrt(3), std * np.sqrt(3)) # uniform distribution 식.
```

* Xavier initialization 과 함께 쓰면 좋은 activation function: `tanh`
  * Xavier initialization을 정할 때, 저자들은 한가지 가정을 더 했는데, activation function이 `(0, 0)` 에서 symmetric 하다는 것임.
  * 따라서 `tanh` 를 activation function으로 쓸 때에는 Xavier initialization이 좋으나, `sigmoid` 에서는 성능이 별로 좋지 못했음.
  * `sigmoid` activation function의 경우 function의 결과가 모두 양수라서 layer의 갯수가 많을 때는 학습이 잘 안되는 경향이 있다고 함.
  * 이 논문이 발표될 당시는 `ReLU` 가 제안되기 전이기 때문에 `ReLU` 에 대한 검토는 논문에 되어있지 않음.



### 3. LeCun Initializaiton

* Deep한 모델에서 잘 안된다고 하는 말이 있어서 넘어감.



### 4. He Initialization 

* 2015년에 이미지넷 챌린지에서 좋은 성과를 거둔 Kaiming He가 제안한[3] 초기화 방법임. 
  * PReLU를 제안하고, activation function을 PReLU로 바꾼 뒤 initialization을 He initialization으로 바꾼 것만으로 좋은 성과를 거두었음.
* He initialization도 분산을 특정 값으로 맞춰주는 것에 초점을 두고 있음. 
  * 증명과정에서도 아예 [2]번 논문에 이어서 설명하는 것이라는 식으로 설명.
  * [2]논문은 증명과정에서 activation function이 zero mean을 가진다고 가정했는데, ReLU는 그렇지 아니하므로 증명과정에서 발생하는 차이가 있고 해당 부분을 아는 것이 논문 증명 이해에 큰 도움이 됨.
  * [2] 논문은 activiation function을 linear 하다고 가정했는데, [3] 논문은 ReLU(혹은 PReLU로까지 확장함)를 가정해서 activiation function 자체에서의 엄밀함을 더 추구함.
* He initialization이 추구하는 분산은 간단히 말하면, Xavier initialization이 추구하는 분산의 2배임.
  * ReLU가 음수 쪽 신호를 완전히 없애버린다는 것에서 착안해, Xavier initialization이 만들어 내는 분산보다 두 배 큰 분산을 만들어 내야 결국 분산이 유지된다고 본 것.
  * 정확한 값은 아래와 같음.
    * 주목할 만한 점은 Xavier initialization과 다르게, `fan_out` 의 값은 사용하지 않음.
    * [3] 논문의 6번 식을 보면 왜 `fan_out` 값이 필요 없는 지 이해할 수 있음.

```python
Var[w] = 2 / fan_in
```

* PReLU(=leaky ReLU)의 경우에는 분산을 다음과 같이 맞춰야 함.
  * 식 15에서 알 수 있고, 증명 과정도 논문에 나와있음.

```
Var[w] = 2 / (fan_in * (1 + a^2))
```

* 역시 He initialization도 분산만 맞추면 되므로, uniform인지 normal인지는 별로 중요치 않음. 
  * Keras implementation도 그래서 두 가지 버전이 존재.
    * uniform: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/he_uniform
    * normal: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/he_normal
  * 다만 Xavier initialization에서 설명한 것과 마찬가지 논리로 uniform distribution이 더 좋을 것으로 예상.
* 실험 성능: Layer가 굉장히 많은 경우(22개)가 아니면, Xavier Initialization과 비슷비슷함.



### 5. Bias Initialization

* 지금까지 `sigma(Wx + b)` 의 꼴에서 W를 어떻게 초기화 할지에 대해 알아보았음.
* 그러면, Bias는 어떻게 초기화 할까?
* ReLU를 쓸 때에는 그냥 0으로 초기화 한다고 함[8].
* 여기저기서 모두 0으로 초기화 하고 있다고 하는데, 왜 0으로 초기화 해야 하는 지는 잘 모르겠음.
  * [8]에 따르면, "ReLU의 특성 상 처음부터 아주 많은 값들이 살아남는 결과로 이어질 수 있기 때문에 그냥 0을 쓰는 것이 좋다"고 함.



### 6. Further

* 물론 더 좋은 initialization 방법도 많다고 함. 
  * CS231n 유투브 비디오 끝에서 읽어볼만한 논문 몇 개 더 소개 됨.
  * 그 중에 간단하면서도 퍼포먼스가 좋은 초기화 방법만 정리해 본 것임.



## 관련 논문 

[1] Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets." Neural computation 18.7 (2006): 1527-1554. 

<https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf> 

[2] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." Proceedings of the thirteenth international conference on artificial intelligence and statistics. 2010. 

http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi 

[3] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." Proceedings of the IEEE international conference on computer vision. 2015.

https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf



## 참고 자료

[4] <https://www.youtube.com/watch?v=GUtlrDbHhJM>, 36:58 ~ 51:19, 슬라이드: http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture06.pdf , 실라버스: http://cs231n.stanford.edu/syllabus.html

[5] <https://www.youtube.com/watch?v=4rC0sWrp3Uw&t=5s> 

[6] https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

[7] <https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network> , 참고: 잘못된 정보가 있는 것 같음.

[8] <http://www.khshim.com/archives/641> 

[9] http://www.jefkine.com/deep/2016/08/08/initialization-of-deep-networks-case-of-rectifiers/

[10] https://github.com/denizyuret/Knet.jl/pull/311

