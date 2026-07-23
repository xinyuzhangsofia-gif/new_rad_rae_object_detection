Train sequences: (1, 5, 6, 14, 15, 18, 20)
Evaluation scope: full
Selection rule: for each model and each test sequence separately, choose one epoch that maximizes the joint sequence score `bev@0.3 + 3d@0.3`, then report both metrics from that same epoch.

Empty-frame definition: frames with zero Sedan GT boxes in the test sequence.
Empty rate definition: empty_frames / test_frames.

| Test Seq | Test Sedan BBox | Empty Frames | Empty Rate | Model7 BEV AP | Model8 BEV AP | Model15 BEV AP | Model7 3D AP | Model8 3D AP | Model15 3D AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 1966 | 375 | 31.09% | 19.5885 (e16) | 22.7238 (e6) | 37.9226 (e13) | 8.0891 (e16) | 8.2847 (e6) | 15.7056 (e13) |
| 10 | 2482 | 263 | 21.86% | 33.1921 (e18) | 27.4785 (e5) | 19.3472 (e5) | 12.0352 (e18) | 8.8939 (e5) | 6.9035 (e5) |
| 11 | 3838 | 63 | 5.27% | 19.6260 (e16) | 23.9459 (e6) | 30.8500 (e13) | 7.9050 (e16) | 12.0952 (e6) | 13.7110 (e13) |
| 12 | 2001 | 410 | 34.40% | 23.9371 (e13) | 30.1556 (e6) | 28.4903 (e29) | 11.9761 (e13) | 16.7629 (e6) | 13.1673 (e29) |

Notes:
- Model7 source files: `evaluation_plots/model7_64_128_ig_train_1,5,6,14,15,18,20_lr5e-5/*.txt`
- Model8 source files: `evaluation_plots/model8_128_128_ig_train_1,5,6,14,15,18,20_lr1e-4/*.txt`
- Model15 source files: `evaluation_plots/model15_128_128_ig_train_1,5,6,14,15,18,20_lr1e-4/*.txt`
- This table supersedes the earlier single-epoch summary, which used the wrong rule.
