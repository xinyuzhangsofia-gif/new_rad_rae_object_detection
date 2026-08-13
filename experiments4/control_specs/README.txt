Test-domain control specifications

Enumerate all sequence_information.csv normal-weather sequences with the target road type; reject missing data and every frame used by any source-trained checkpoint in the weather group; rank remaining continuous candidates with train_cfg total/range/empty control priorities.

Selected fixed controls:
- heavy_snow_test_46_47: target seq46_47 <- normal source seq11: frames=1195, kept_bbox=434, masked_bbox=3405, deficit=0
- heavy_snow_test_54_55: target seq54_55 <- normal source seq1: frames=597, kept_bbox=1204, masked_bbox=0, deficit=480
- heavy_snow_test_54_56: target seq54_56 <- normal source seq1: frames=597, kept_bbox=1204, masked_bbox=0, deficit=266
- heavy_snow_test_55_56: target seq55_56 <- normal source seq1: frames=597, kept_bbox=1204, masked_bbox=0, deficit=46
- light_snow_test_42: target seq42 <- normal source seq5: frames=598, kept_bbox=970, masked_bbox=1687, deficit=0
- light_snow_test_43: target seq43 <- normal source seq5: frames=598, kept_bbox=1335, masked_bbox=1322, deficit=0
- light_snow_test_48: target seq48 <- normal source seq11: frames=606, kept_bbox=362, masked_bbox=1310, deficit=0
- light_snow_test_49: target seq49 <- normal source seq11: frames=608, kept_bbox=139, masked_bbox=1540, deficit=0
- overcast_test_13: target seq13 <- normal source seq10: frames=1190, kept_bbox=305, masked_bbox=2161, deficit=0
- overcast_test_22: target seq22 <- normal source seq18: frames=594, kept_bbox=1444, masked_bbox=161, deficit=0
- rain_test_23: target seq23 <- normal source seq18: frames=594, kept_bbox=1570, masked_bbox=35, deficit=0
- rain_test_24: target seq24 <- normal source seq18: frames=594, kept_bbox=1605, masked_bbox=0, deficit=426
- rain_test_25: target seq25 <- normal source seq18: frames=594, kept_bbox=1605, masked_bbox=0, deficit=416
- sleet_test_50: target seq50 <- normal source seq11: frames=597, kept_bbox=1077, masked_bbox=780, deficit=0
- sleet_test_51: target seq51 <- normal source seq11: frames=597, kept_bbox=260, masked_bbox=1597, deficit=0
- sleet_test_52: target seq52 <- normal source seq11: frames=597, kept_bbox=438, masked_bbox=1419, deficit=0
- sleet_test_53: target seq53 <- normal source seq11: frames=597, kept_bbox=590, masked_bbox=1267, deficit=0

Each weather_test_*/test.txt uses strict sequence,frame.txt syntax. Each object_ignore_override.json makes surplus source Sedan boxes neutral (neither positives nor false-positive background).

See index.json and each stats.json/comparison.txt for the complete candidate ranking, post-FOV counts, target quartiles, and leakage audit.
