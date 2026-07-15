Train sequences: (1, 5, 6, 14, 15, 18, 20)
Evaluation scope: full
Selection rule: choose one representative epoch for each model by averaging all available `bev@0.3` and `3d@0.3` values across test sequences, then pick the epoch with the highest combined mean.
Selected epochs:
- model7: epoch 13, mean_bev@0.3 = 22.9686, mean_3d@0.3 = 7.5452, combined_mean = 15.2569
- model8: epoch 6, mean_bev@0.3 = 23.5518, mean_3d@0.3 = 10.6204, combined_mean = 17.0861
- model15: epoch 13, mean_bev@0.3 = 26.2808, mean_3d@0.3 = 10.7482, combined_mean = 18.5145

Empty-frame definition: frames with zero Sedan GT boxes in the test sequence.
Empty rate definition: empty_frames / test_frames.

| Test Seq | Test Sedan BBox | Empty Frames | Empty Rate | Model7 BEV AP(epoch 13) | Model8 BEV AP(epoch 6) | Model15 BEV AP(epoch 13) | Model7 3D AP(epoch 13) | Model8 3D AP(epoch 6) | Model15 3D AP(epoch 13) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 1966 | 375 | 31.09% | 13.3823 | 22.7238 | 37.9226 | 2.7205 | 8.2847 | 15.7056 |
| 10 | 2482 | 263 | 21.86% | 34.6082 | 17.3819 | 10.0699 | 9.2015 | 5.3389 | 2.8279 |
| 11 | 3838 | 63 | 5.27% | 19.9466 | 23.9459 | 30.8500 | 6.2826 | 12.0952 | 13.7110 |
| 12 | 2001 | 410 | 34.40% | 23.9371 | 30.1556 | - | 11.9761 | 16.7629 | - |

Notes:
- Model7 source files: `evaluation_plots/model7_64_128_ig_train_1,5,6,14,15,18,20/*.txt`
- Model8 source files: `evaluation_plots/model8_128_128_ig_train_1,5,6,14,15,18,20/*.txt`
- Model15 source files: `evaluation_plots/model15_128_128_ig_train_1,5,6,14,15,18,20/*.txt`
- `model15` currently has no `val_seq_12` txt under `evaluation_plots`, so seq12 stays blank.
- The model15 representative epoch was selected using the available seq9, seq10, seq11 results only.
