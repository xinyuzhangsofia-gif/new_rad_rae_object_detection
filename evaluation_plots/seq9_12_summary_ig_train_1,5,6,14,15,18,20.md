Train sequences: (1, 5, 6, 14, 15, 18, 20)
Evaluation scope: full
Empty-frame definition: frames with zero Sedan GT boxes in the test sequence.
Empty rate definition: empty_frames / test_frames.

| Test Seq | Test Sedan BBox | Empty Frames | Empty Rate | Model7 BEV AP(best) | Model8 BEV AP(best) | Model15 BEV AP(best) | Model7 3D AP(best) | Model8 3D AP(best) | Model15 3D AP(best) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 1966 | 375 | 31.09% | 20.0602 (e29) | 23.2361 (e10) | 37.9226 (e13) | 8.0891 (e16) | 8.2847 (e6) | 15.7056 (e13) |
| 10 | 2482 | 263 | 21.86% | 34.6082 (e13) | 27.4785 (e5) | 19.3472 (e5) | 12.0352 (e18) | 10.5158 (e9) | 7.6770 (e2) |
| 11 | 3838 | 63 | 5.27% | 20.7641 (e19) | 23.9459 (e6) | 30.8500 (e13) | 8.1229 (e17) | 12.0952 (e6) | 14.4190 (e18) |
| 12 | 2001 | 410 | 34.40% | 23.9371 (e13) | 30.1556 (e6) | 28.4903 (e29) | 11.9761 (e13) | 16.7629 (e6) | 13.2779 (e19) |

Notes:
- Model7 source: `evaluation_plots/model7_64_128_ig_train_1,5,6,14,15,18,20/*.txt`
- Model8 source: `evaluation_plots/model8_128_128_ig_train_1,5,6,14,15,18,20/*.txt`
- Model15 source: `evaluation_plots/model15_128_128_ig_train_1,5,6,14,15,18,20/*.txt`
- In this table, each BEV/3D entry keeps its own best epoch, so epochs inside the same row can differ between BEV and 3D.
