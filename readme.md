# The code for our newest paper "Difference Decomposition Networks for Infrared Small Target Detection"
The paper can be availble in <https://arxiv.org/abs/2512.03470>.
In addition, the code for STD2Net could be accessible in [STD2Net's code](https://github.com/greekinRoma/STD2Net).
## 1. Backbone 
Our backbone is a structure with 3-level's structure, as shown below.

![The 3-level's structure.](3UNet.pdf)

## 2. Basis Decomposition
As we all know the basis decomposition is fundamental math operation, which could decompose the origin feature into a group of elements.
![Basis decomposition theory](Basis_decomposition_theory.pdf)
### 2.1 Difference Basis Decomposition
Based on the Basis Decomposition, we propose the Difference Basis Decomposition for Infrared Small Target Detection (IRSTD), by difference element's decomposing the origin features and enhancing themselves. 
![Difference Basis Decomposition](Decomposition.pdf)
### 2.2 Spatial Difference Basis Decomposition
For the infrared targets' spatial features, we utilize the Spatial Difference Decomposition Module (SD2M) and Spatial Difference Decomposition Downsamplinng (SD2D).
![SD2M](SDecM.pdf)
![SD2D](SDecD.pdf)
### 2.3 Temporal Difference Basis Decomposition
For the infrared targets' temporal features, we utilize the Temporal Difference Decomposition Module (TD2M).
![TD2M](TDecM.pdf)
## 3. Reference
```
@misc{hu2026differencedecompositionnetworksinfrared,
      title={Difference Decomposition Networks for Infrared Small Target Detection}, 
      author={Chen Hu and Mingyu Zhou and Shuai Yuan and Hongbo Hu and Zhenming Peng and Tian Pu and Xiying Li},
      year={2026},
      eprint={2512.03470},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.03470}, 
}
```