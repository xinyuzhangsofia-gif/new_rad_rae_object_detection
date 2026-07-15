# domain_shift_plan_s_same_test_5_14

This split stores the file-based split for **plan S**.

## Meaning

- `test` stays fixed as the same target-domain test set: `sequence 5` + `sequence 14`
- `train` uses the second source-domain pool with continuous frame windows
- line format is the same as the current project file split: `sequence,frame_name.txt`

## Train definition

- `sequence 2`: full sequence
- `sequence 9`: first 600 frames
- `sequence 10`: first 600 frames
- `sequence 11`: last 600 frames
- `sequence 12`: first 600 frames

## Test definition

- `sequence 5`: full sequence
- `sequence 14`: full sequence

## Train details

- seq2: 598 frames, 00031 ... 00628, empty=500, sedan=1, bus=98, bbox=99
- seq9: 600 frames, 00024 ... 00623, empty=0, sedan=1508, bus=369, bbox=1877
- seq10: 600 frames, 00027 ... 00626, empty=0, sedan=1764, bus=310, bbox=2074
- seq11: 600 frames, 00629 ... 01228, empty=7, sedan=1869, bus=477, bbox=2346
- seq12: 600 frames, 00037 ... 00636, empty=1, sedan=1613, bus=600, bbox=2213

**Total**: frames=2998, empty=508, sedan=6755, bus=1854, bbox=8609

## Test details

- seq5: 598 frames, 00032 ... 00629, empty=1, sedan=2625, bus=0, bbox=2625
- seq14: 595 frames, 00034 ... 00628, empty=273, sedan=322, bus=0, bbox=322

**Total**: frames=1193, empty=274, sedan=2947, bus=0, bbox=2947

## How to use

```python
"split_mode": "file",
"split_dir": "split/domain_shift_plan_s_same_test_5_14",
```
