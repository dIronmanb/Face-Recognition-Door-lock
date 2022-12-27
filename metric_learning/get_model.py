from ResNext import ResNext
from ResNext import Bottleneck

def resnext50():
    return ResNext(Bottleneck, [3,4,6,3], vector_dim=512, groups=4, width_per_group=32)

def resnext101():
    return ResNext(Bottleneck, [3,4,23,3])

def resnext152():
    return ResNext(Bottleneck, [3,8,36,3])
