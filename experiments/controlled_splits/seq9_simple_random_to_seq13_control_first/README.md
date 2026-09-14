# seq9 simple random to seq13 control (historical pre-generated asset)

This is the simplified version:
- choose one continuous seq9 window
- count seq13 sedan/bus bbox in ridx bins `(0-80)` and `(80-144)`
- randomly keep seq9 objects to match those counts
- convert all other seq9 sedan/bus GT to ignore
- choose the random seed whose empty-frame count is closest to seq13

## Selected window

- source sequence: `9`
- target sequence: `13`
- window position: `first`
- start_file_idx: `0`
- end_file_idx: `1189`
- start_frame_name: `00024`
- end_frame_name: `01213`
- selected random seed: `291`

## Use

```python
"split_mode": "kradar_file",
"split_dir": "experiments/controlled_splits/seq9_simple_random_to_seq13_control_first",
"gt_object_ignore_override_path": "experiments/controlled_splits/seq9_simple_random_to_seq13_control_first/object_ignore_override.json",
```

## Summary

- target empty/non-empty: `974/216`
- achieved empty/non-empty: `877/313`
- target boxes in bins: `389`
- achieved boxes in bins: `389`
