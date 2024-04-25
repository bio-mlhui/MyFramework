## Task-Agnostic Coding Framework Overview

This is my task-agnostic coding framework. It currently supports various functionalities including:

- **Referring Video/Image Segmentation (RIS/RVOS)**
- **Image/Video Instance/Object Segmentation (VIS/VOS)**

### Ongoing Developments

I am now working on incorporating 3D modalities into the framework:

- **Novel-View Synthesis (Images to 3D)**
- **Dynamic Novel-View Synthesis (Video to 4D)**

### Current Version

This codebase includes implementations of the following papers:

1. [ECCV2024 submission](./readme/EventRR.pdf)  [CVPR2024 reviews](./readme/screencapture-openreview-net-forum-2024-04-25-11_37_54.png) 
    - **Title:** "EventRR: Event Referential Reasoning for Referring Video Object Segmentation"
    - **Authors:** Huihui Xu, [Qiang Nie], Lei Zhu
    - **[Main Code: Temporal Concept-Role Reasoning](./models/graph/reason_module.py)**

2. [MICCAI2024 submission](./readme/LGRNet.pdf)
    - **Title:** "LGRNet: Local-Global Reciprocal Network for Uterine Fibroid Segmentation in Ultrasound Videos"
    - **Authors:** Huihui Xu, [Yijun Yang (dataset contribution), Angelica Aviles-Rivero, Guang Yang, Jing Qin], Lei Zhu
    - **[Main Code: Hilbert Selective Scan](./models/encoder/ops/modules/frame_query_ss2d.py#L576)**

3. I've implemented the **[Deformable Selective Scan](./models/encoder/ops/modules/deform_selective_scan_mamba_scan.py)**, and achieved SOTA(IoU=65.8, BER=7.88) on the **Video Shadow Detection(VSD)** task. But I plan to first work on the 3D neural rendering task. A paper on VSD will be submitted to TCSVT after NIPS ddl.

