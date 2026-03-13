## *Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs* 
### (NeurIPS'25 - https://arxiv.org/abs/2510.17364)
---
![rLiVS](figs/teaser.jpg)
Steps
---
- create conda env and install requirements (check below)
- dowload data and insert the correct paths in the files
- Pipeline for a benchmark includes (check below):
   - (a) processing the videos and extracting memories ("...extract..." files)
   - (b) answering the questions based on the extracted memories ("...retrieve_and_answer..." files)

Env
---
- pip install torch==2.2.2 torchvision torchaudio
- pip install transformers==4.47.0
- (pip uninstall flash-attn -y)
- find flash_attn correct version based on python/torch/cuda: https://mjunya.com/flash-attention-prebuild-wheels/?flash=2.7.3&python=3.10&torch=2.2&cuda=12.1

**copy-paste modeling_qwen2_4_47_0.py under transformers (replace the defaults modeling file - works for the specific version: 4.57.0)**

*main changes:* (to enable layer specific attn_implementation)
- qwen_model: attention_mask based on layer_idx
- decoder_layer: pass output_attentions = T or F to attn based on layer_idx
- attn: flash -> if output_attentions = T fallback to eager

Run
---
- *extraction*: python -u llava_ov_extract_realtime_recurrent_simple.py --dataset movienet
- *retrieval & answering*: python -u llava_ov_retrieve_and_answer_movienet_realtime.py
---
## Copyright
Copyright (c) 2026 Toyota Motor Europe. All rights reserved.

Patent pending.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]. See the LICENSE file for details.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
