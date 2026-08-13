Source-domain performance difference for Heavy Snow, Light Snow, Overcast, Rain, and Sleet.

Every AP is official revised K-Radar AP@IoU 0.3 averaged over epochs 5-24 of the same source-trained checkpoint. BEV_normal/3D_normal use the controlled normal test; BEV_weather/3D_weather reuse/recompute that checkpoint's adverse-weather target-test result with identical settings. SD = AP_normal - AP_weather, so positive SD denotes adverse-weather degradation. q1-q4 use the target-weather GT distance boundaries for both domains.

average_all_valid includes every completed valid control, including reported source-GT deficits. average_exact includes only controls whose normal and weather eligible Sedan totals match exactly.
