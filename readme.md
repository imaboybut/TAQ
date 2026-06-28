python inference.py --lq_root sample/018 --quant_state_path TAQ_8bit.pth --save_root results --device cuda:0 --interval 100
python make_video.py --input_dir results/018 --out results/video/018.mp4 --fps 24



python make_video.py --input_dir sample/018 --out results/video/sample_018.mp4 --fps 24
