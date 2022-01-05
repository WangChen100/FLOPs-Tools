# FLOPs-Tools

This is a tool for calculating FLOPs of torch models.





## 文件组成

------- FLOPs-Tools 

 |----- calculate_flops.py  **用于计算FLOPs的例子** 

 |----- compute_flops.py  用于计算FLOPs和运行速度

 |----- jit_handles.py     用于计算各个操作的flops

 |----- __ init __.py

 |----- README.md



## 工具用法

在自己的的代码中，初始化一个输入张量和代测模型，调用

```python
flop_count(model, (intput_tensor,)
```

返回包含计算量的字符串，打印即可





### 例子

```python
import torch
from extra.flop_count import flop_count
from models.modules.basic_block import BasicBlock 

planes = 32
input_size=[1, planes, 96, 72]

dump_input = torch.rand(input_size)

basicblock = BasicBlock(planes,planes)

print("FLOPs of basicblock is: ")
print(flop_count(basicblock, (dump_input,)))

```

