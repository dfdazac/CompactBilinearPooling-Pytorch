# CompactBilinearPooling-Pytorch

A Pytorch Implementation for Compact Bilinear Pooling. Adapted from [tensorflow_compact_bilinear_pooling](https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling)
Updated for PyTorch 0.4.

## Usage

```python
import torch
from CompactBilinearPooling import CompactBilinearPooling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bottom1 = torch.randn(128, 512, 14, 14).to(device)
bottom2 = torch.randn(128, 512, 14, 14).to(device)

layer = CompactBilinearPooling(512, 512, 8000).to(device)

out = layer(bottom1, bottom2)
```


## Reference

```
Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (2016).
Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).
```
