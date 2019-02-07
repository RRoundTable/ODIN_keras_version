# ODIN_keras_version
Out-of-distribution을 검출하는 방법론 중 하나인 odin(pytorch)을 keras화 하였습니다.


# ODIN

out-of-distribution을 검출하는 Odin의 장점은 pretrained된 model을 그대로 사용할 수 있다는 점입니다.

기존의 pretrained model에 추가적으로 T scaling과 perturbation을 적용하여,

in-distribution과 out-of-distribution을 구분할 수 있습니다.

reference : https://github.com/facebookresearch/odin

