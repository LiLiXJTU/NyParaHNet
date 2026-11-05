# NyParaHNet

NyParaHNet is a medical image segmentation architecture based on NyPara blocks.

The study evaluates segmentation performance on Synapse, ACDC, FIVES, and AMOS2022.

## Repository status

This repository currently contains a project overview only. Training and inference code, environment specifications, dataset preparation instructions, and pretrained weights are not yet included.

Reproduction instructions will need to accompany a code release; the current repository is not a runnable implementation.

## Baseline official repositories and project links

The following list provides upstream project sources for the comparison methods. Links were checked on November 5, 2026. These are source references; the exact commits, local adaptations, and configurations used in the NyParaHNet experiments are not yet documented here.

### 2D segmentation baselines

| Method | Official repository or project source | Notes |
| --- | --- | --- |
| U-Net | [Original authors' project and software release](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) | Original Caffe-based release; this is not a claim that the original Caffe implementation was used in our PyTorch experiments. |
| TransUNet | [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) | Authors' implementation. |
| Swin-Unet | [HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) | Authors' implementation. |
| MISSFormer | [ZhifangDeng/MISSFormer](https://github.com/ZhifangDeng/MISSFormer) | Authors' implementation. |
| TransDeepLab | [Repository announced in the paper](https://github.com/rezazad68/transdeeplab) · [Paper](https://arxiv.org/abs/2208.00713) | The announced repository returned HTTP 404 when checked; no third-party replacement is labeled as official here. |
| HiFormer | [amirhossein-kz/HiFormer](https://github.com/amirhossein-kz/HiFormer) | Authors' implementation. |
| TransCASCADE | [SLDGroup/CASCADE](https://github.com/SLDGroup/CASCADE) | Official CASCADE implementation containing TransCASCADE. |
| VM-UNet | [JCruan519/VM-UNet](https://github.com/JCruan519/VM-UNet) | Official VM-UNet implementation. |
| PVT-GCASCADE | [SLDGroup/G-CASCADE](https://github.com/SLDGroup/G-CASCADE) | Official G-CASCADE implementation; PVT-GCASCADE uses the PVT encoder. |
| UCFilTransNet | [LiLiXJTU/UCFilTransNet](https://github.com/LiLiXJTU/UCFilTransNet) | UCFilTransNet project repository. |
| D-LKA-Former / D-LKA Net | [xmindflow/deformableLKA](https://github.com/xmindflow/deformableLKA) | Implementation of *Beyond Self-Attention: Deformable Large Kernel Attention for Medical Image Segmentation*; includes 2D and 3D code. |
| CSWin-UNet | [eatbeanss/CSWin-UNet](https://github.com/eatbeanss/CSWin-UNet) | Repository linked in the authors' paper. |
| CFFormer | [JiaxuanFelix/CFFormer](https://github.com/JiaxuanFelix/CFFormer) | Medical image segmentation CFFormer; distinct from the similarly named remote-sensing model. |

### 3D segmentation baselines

| Method | Official repository or project source | Notes |
| --- | --- | --- |
| nnFormer | [282857341/nnFormer](https://github.com/282857341/nnFormer) | Volumetric medical image segmentation implementation. |
| UNETR | [Project-MONAI/research-contributions: UNETR/BTCV](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV) | UNETR implementation in MONAI Research Contributions. |
| D-LKA-Former / D-LKA Net | [xmindflow/deformableLKA: 3D](https://github.com/xmindflow/deformableLKA/tree/main/3D) | 3D implementation in the authors' repository. |
| EffiDec3D | [SLDGroup/EffiDec3D](https://github.com/SLDGroup/EffiDec3D) | Authors' implementation. |
| MixUNETR | [skyous779/MixUNETR](https://github.com/skyous779/MixUNETR) | Authors' implementation for prostate MRI segmentation. |

### Sources for additional ablation comparisons

| Component | Original repository | Scope |
| --- | --- | --- |
| Nystromformer | [mlpen/Nystromformer](https://github.com/mlpen/Nystromformer) | Original attention implementation; the adaptation to the segmentation encoder must be documented separately. |
| Global Filter Network (GFNet) | [raoyongming/GFNet](https://github.com/raoyongming/GFNet) | Reference for learnable Fourier-domain filtering in the FFT-based comparison; not a claim that the complete GFNet classifier was used as a segmentation baseline. |

Please cite the respective original papers and follow each source's license when reusing its code.
