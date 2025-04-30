# Activation-Aware Bitdelta

A contribution by to [[BitDelta](https://github.com/FasterDecoding/BitDelta)] that allows initializing scale factors using activation statistics, avoiding end-to-end distillation. This can be be enabled by adding the `--use_activation_aware` flag. Achieves 1.5-2x faster wall clock time with either minimal loss or an improvement in accuracy.
