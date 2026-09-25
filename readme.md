# TAQ: Static-Deployable Temporal-Aware Quantization for Real-World Video Super-Resolution

**ECCV 2026** · Official PyTorch implementation

Jinwoo Chung, Sangho An, Sungyeop Jung, Jangho Kim
Kookmin University

[**Paper (ECCV 2026)**](https://eccv.ecva.net/virtual/2026/poster/5539)

## Highlights

- **Static PTQ for video SR.** TAQ uses video structure only during offline calibration and outputs one fixed set of quantization parameters. No range or scale is recomputed at inference, so the model compiles directly to static INT8 engines.
- **Real speedup on the edge.** With TensorRT on an NVIDIA Jetson Orin Nano, TAQ INT8 runs **3.56× faster** than FP32, while a dynamic INT8 baseline is slower than FP32 (0.79×).
- **Better temporal stability.** Sequence-wise calibration and a temporal-consistency objective reduce flicker and drift, improving LPIPS by up to 0.0334 and TLPIPS by up to 4.7344 over static PTQ baselines under identical calibration.

## Method

TAQ calibrates a pre-trained VSR model in three stages and keeps every quantizer a plain uniform affine quantizer:

1. **Sequence-wise histogram initialization.** Each calibration sequence is processed on its own. Activations are percentile-clipped (0.1% / 99.9%), accumulated into a 1024-bin histogram, and the bounds (ℓ, u) that minimize the histogram reconstruction error are installed. Weight bounds minimize the MSE between each weight tensor and its quantized version.
2. **Temporal refinement of quantization bounds.** With the weights frozen, the bounds of every weight and activation quantizer are refined by Adam so that the inter-frame differences of the quantized model match those of the floating-point model.
3. **Sequence-wise bounds ensembling.** The refined per-sequence bounds are averaged into one deployable set and written back into the quantizers.

## Results

RealViformer ×4 on REDS. PSNR↑ / SSIM↑ / LPIPS↓ / TLPIPS↓ / TOF↓, all methods calibrated on the same 9 REDS sequences (20 frames each).

| Bit | Metric | MinMax | Percentile | PTQ4SR | 2DQuant | **TAQ (ours)** |
|:---:|:---|---:|---:|---:|---:|---:|
| 4 | PSNR↑ | 10.7379 | 23.1268 | 23.3999 | 23.1155 | **23.8949** |
| 4 | SSIM↑ | 0.1440 | 0.4689 | 0.5329 | 0.5306 | **0.5480** |
| 4 | LPIPS↓ | 0.8517 | 0.5555 | 0.6793 | 0.4828 | **0.4724** |
| 4 | TLPIPS↓ | 36.1503 | 15.3454 | 18.9417 | 12.9041 | **10.0249** |
| 4 | TOF↓ | 1385.4387 | 55.4030 | 29.4118 | 29.0666 | **23.6122** |
| 8 | PSNR↑ | 24.9587 | 25.4449 | 25.7334 | 25.7180 | **25.7768** |
| 8 | SSIM↑ | 0.6675 | 0.6493 | 0.6694 | 0.6767 | **0.6772** |
| 8 | LPIPS↓ | 0.2566 | 0.3122 | 0.2856 | 0.2660 | **0.2517** |
| 8 | TLPIPS↓ | 1.8291 | 1.9872 | 2.7159 | 1.4566 | **1.2161** |
| 8 | TOF↓ | 6.4111 | 6.3244 | 12.0487 | 6.9301 | **5.9729** |

TensorRT deployment on NVIDIA Jetson Orin Nano (BasicVSR backbone, 32 frames from Vid4):

| Method | Precision | Throughput (fps)↑ | Latency (s)↓ | Speedup↑ |
|:---|:---|---:|---:|---:|
| BasicVSR | FP32 | 0.3245 | 2.987 | – |
| QBasicVSR | INT8 (dynamic) | 0.2661 | 3.738 | 0.79× |
| **TAQ (ours)** | INT8 (static) | **1.1459** | **0.839** | **3.56×** |

Full comparisons at 2/3/4/8 bit on REDS, SPMCS, UDM10 and VideoLQ, the RealBasicVSR backbone, and the QBasicVSR comparison are in the paper.

## Visual Results

### REDS

<table>
  <tr>
    <th align="center">Low-Quality Input</th>
    <th align="center">TAQ (ours)</th>
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
    <th align="center">TAQ (ours)</th>
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

## Getting Started

### Installation

Python ≥ 3.10 and a CUDA build of PyTorch ≥ 2.0.

```bash
git clone https://github.com/imaboybut/TAQ.git
cd TAQ
pip install -r requirements.txt
```

### Pretrained model

Calibration starts from the official FP32 [RealViformer](https://github.com/Yuehan717/RealViformer) weights (`weights.pth`).

### Calibration data

The paper calibrates on **REDS validation clips 021–029, first 20 frames each** (9 sequences, 180 frames), degraded with the RealBasicVSR real-world degradation pipeline. Any directory of LQ frame folders works, one sub-folder per sequence:

```text
calib/
├── 021/00000000.png, 00000001.png, ...
├── 022/
└── ...
```

### Calibration

`calibrate.py` runs the three stages end to end and exports one static checkpoint. The defaults are the paper setting, so only the bit-width has to be chosen. Calibration is cheap (about 30 minutes for the 9 REDS clips on one A6000), so it is straightforward to calibrate for your own calibration set, bit-width or test data.

```bash
python calibrate.py \
  --model_path pretrained_model/weights.pth \
  --calib_root /path/to/REDS/val_lq_realbasic/calib \
  --calib_sequences 021,022,023,024,025,026,027,028,029 \
  --w_bit 8 --a_bit 8 \
  --out_dir calib_out/8bit \
  --save_path calib_out/8bit/TAQ_8bit.pth \
  --device cuda:0
```

`--out_dir` receives the refined bounds of each sequence (`<seq>_frames0-20_ranges.json`) and the ensembled bounds (`averaged_ranges.json`); `--save_path` is the static checkpoint used for inference.

| Option | Default | Description |
|:---|:---:|:---|
| `--w_bit`, `--a_bit` | 8, 8 | Weight / activation bit-width (the paper reports 2, 3, 4 and 8) |
| `--calib_frames` | 20 | Frames per sequence used for initialization and refinement |
| `--range_ft_epochs`, `--range_ft_lr` | 5, 5e-4 | Temporal refinement epochs per sequence and Adam learning rate |
| `--calib_clip_low`, `--calib_clip_high` | 0.001, 0.999 | Percentile clipping applied to activations before the histogram (`--disable_calib_clip` turns it off) |
| `--skip_io_layers` | off | Keep the first and last convolution in FP32. By default all 204 Conv/Linear layers of RealViformer are quantized, as in the paper |

### Inference

```bash
python inference.py \
  --lq_root /path/to/lq_frames \
  --quant_state_path calib_out/8bit/TAQ_8bit.pth \
  --save_root results \
  --device cuda:0 --interval 100
```

`--lq_root` may point to one folder of LQ frames or to a directory of such folders (one output folder per sequence). `--interval` sets how many frames are processed per forward pass.

## Repository Structure

```text
.
├── archs/               # RealViformer architecture (official release)
├── calibration.py       # histogram observer and MSE bound search
├── quant_layers.py      # affine fake quantizer, QuantConv2d / QuantLinear
├── TAQ.py               # quantized-model wrapper: layer injection, activation capture, range I/O
├── calibrate.py         # end-to-end calibration and static checkpoint export
├── inference.py         # inference with an exported static checkpoint
├── data_util.py, img_util.py
└── result/              # demo videos
```

## Citation

```bibtex
@inproceedings{chung2026taq,
  title={TAQ: Static-Deployable Temporal-Aware Quantization for Real-World Video Super-Resolution},
  author={Chung, Jinwoo and An, Sangho and Jung, Sungyeop and Kim, Jangho},
  booktitle={European Conference on Computer Vision},
  pages={167--184},
  year={2026},
  organization={Springer}
}
```

## Acknowledgements

The RealViformer architecture and data utilities come from the official [RealViformer](https://github.com/Yuehan717/RealViformer) release. The histogram-based bound search follows the MSE calibration of [2DQuant](https://github.com/Kai-Liu001/2DQuant). Image utilities are adapted from [BasicSR](https://github.com/XPixelGroup/BasicSR).
