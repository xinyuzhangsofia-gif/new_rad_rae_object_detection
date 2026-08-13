Rain and sleet checkpoint re-evaluation with official K-Radar AP@IoU 0.3 split by GT-box distance rank quartiles.

Distance is sqrt(x^2+y^2+z^2) at each GT box's radar-frame center. Quartile boundaries and N_bbox are derived only from ground-truth boxes; predictions are filtered using those fixed bounds. Every AP is the mean over epochs 5-24 inclusive. TD = target-trained AP - source-trained AP.
