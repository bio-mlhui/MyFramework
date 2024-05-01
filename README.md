## Task-Agnostic Coding Framework Overview

This is my task-agnostic coding framework. It currently supports various functionalities including:

- **Referring Video/Image Segmentation (RIS/RVOS)**
- **Image/Video Instance/Object Segmentation (VIS/VOS)**

### Ongoing Developments

I am now working on incorporating 3D modalities into the framework:

- **Novel-View Synthesis (Images to 3D)**
- **Dynamic Novel-View Synthesis (Video to 4D)**

### Implemented Papers/Tasks

This codebase includes implementations of the following papers:

1. [ECCV2024 submission](./readme/EventRR.pdf)  [CVPR2024 reviews](./readme/screencapture-openreview-net-forum-2024-04-25-11_37_54.png) 
    - **Title:** "EventRR: Event Referential Reasoning for Referring Video Object Segmentation"
    - **Authors:** Huihui Xu, [Qiang Nie], Lei Zhu
    - **[Main Code: Temporal Concept-Role Reasoning](./models/graph/reason_module.py)**

2. [MICCAI2024 submission](./readme/LGRNet.pdf)
    - **Title:** "LGRNet: Local-Global Reciprocal Network for Uterine Fibroid Segmentation in Ultrasound Videos"
    - **Authors:** Huihui Xu, [Yijun Yang (dataset contribution), Angelica Aviles-Rivero, Guang Yang, Jing Qin], Lei Zhu
    - **[Main Code: Hilbert Selective Scan](./models/encoder/ops/modules/frame_query_ss2d.py#L576)**

3. I've implemented the **[Deformable Selective Scan](./models/encoder/ops/modules/deform_selective_scan_mamba_scan.py)**, and achieved SOTA(IoU=65.8, BER=7.88) on the **Video Shadow Detection(VSD)** task, which is higher thatn CVPR2023 SOTA SCOTCH_AND_SODA. But I plan to first work on the 3D neural rendering task.


### My presentations

1. Final Project of 2022 Fall IOT5501 Convex Optimization (Neural Tangent Kernel) **[Report](./readme/ntk_report.pdf)**  **[Presentation](./readme/ntk_pre.pdf)** 

2. Final Presentation of 2023 Spring ROAS6000G Medical Computer Vision (Using Diffusion for Ambiguous Medical Segmentation) **[Presentation](./readme/ROAS6000G_Presentation.pdf)** 

3. Self Presentation of 2024 Spring ROAS6800E (Introduction of Gaussian Splatting) **[Presentation](./readme/haosen_gs.pdf)**

