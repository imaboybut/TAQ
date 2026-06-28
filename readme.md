# TAQ: Static-Deployable Temporal-Aware Quantization for Real-World Video Super-Resolution

Our paper has been accepted by **ECCV 2026**.

## Overview

**Temporal-Aware Quantization (TAQ)** is a static-deployable post-training quantization framework for real-world video super-resolution.
TAQ uses video structure only during offline calibration and produces a single static quantized model for efficient deployment.

## Visual Results

### REDS

<table>
  <tr>
    <th align="center">Low-Quality Input</th>
    <th align="center">TAQ Ours</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <video src="result/reds_lq_015.mp4" width="100%" controls muted loop></video>
      <br>
      <a href="result/reds_lq_015.mp4">Open video</a>
    </td>
    <td align="center" width="50%">
      <video src="result/reds_ours_015.mp4" width="100%" controls muted loop></video>
      <br>
      <a href="result/reds_ours_015.mp4">Open video</a>
    </td>
  </tr>
</table>

### VideoLQ

<table>
  <tr>
    <th align="center">Low-Quality Input</th>
    <th align="center">TAQ Ours</th>
  </tr>
  <tr>
    <td align="center" width="50%">
      <video src="result/videolq_005.mp4" width="100%" controls muted loop></video>
      <br>
      <a href="result/videolq_005.mp4">Open video</a>
    </td>
    <td align="center" width="50%">
      <video src="result/videolq_ours_005.mp4" width="100%" controls muted loop></video>
      <br>
      <a href="result/videolq_ours_005.mp4">Open video</a>
    </td>
  </tr>
</table>

## Repository Structure

```text
.
├── archs/
├── result/
│   ├── reds_lq_015.mp4
│   ├── reds_ours_015.mp4
│   ├── videolq_005.mp4
│   └── videolq_ours_005.mp4
├── TAQ.py
├── data_util.py
├── img_util.py
├── inference.py
├── make_video.py
├── quant_layers.py
└── README.md
```

## Citation

```bibtex
@inproceedings{chung2026taq,
  title={TAQ: Static-Deployable Temporal-Aware Quantization for Real-World Video Super-Resolution},
  author={Chung, Jinwoo and An, Sangho and Jung, Sungyeop and Kim, Jangho},
  booktitle={European Conference on Computer Vision},
  year={2026}
}
```
