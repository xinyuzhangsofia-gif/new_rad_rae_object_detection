# seq9 matched to seq13 control (historical pre-generated asset)

This directory stores a continuous-frame control split plus an object-level
ignore override. It does not edit the raw K-Radar gt.txt files.

## Selected continuous window

- source sequence: `9`
- target sequence: `13`
- start_file_idx: `16`
- end_file_idx: `1205`
- start_frame_name: `00040`
- end_frame_name: `01229`

## Matching rule

- frame count matches the target sequence exactly
- empty/non-empty frame count matches the target sequence exactly
- sedan/bus bbox totals match the target sequence exactly
- ridx bins are controlled using `(0-80)` and `(80-144)`
- bbox outside those two bins are also tracked under `other` so total counts stay exact
- extra sedan/bus GT are converted to ignore boxes at dataloader time
- kept bbox per frame are capped by the target-sequence max frame count

## Files

- `train.txt`: the selected continuous source window
- `test.txt`: the source frames outside the selected window
- `object_ignore_override.json`: object-level ignore override JSON
- `stats.json`: target, pre-override, and achieved stats

## How to use

Set the following in train/eval/visualize config when you want this control experiment:

```python
"split_mode": "kradar_file",
"split_dir": "experiments/controlled_splits/seq9_matched_to_seq13_control",
"gt_object_ignore_override_path": "experiments/controlled_splits/seq9_matched_to_seq13_control/object_ignore_override.json",
```

If you want to merge these selected seq9 frames into a larger custom split,
reuse `train.txt` entries and keep the same override JSON path.

## Summary

- target empty/nonempty: `974/216`
- achieved empty/nonempty: `974/216`
- target total sedan+bus: `407`
- achieved total sedan+bus: `407`
