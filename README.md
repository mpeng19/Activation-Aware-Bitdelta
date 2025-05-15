# Activation-Aware Bitdelta

A contribution to [[BitDelta](https://github.com/FasterDecoding/BitDelta)] that allows initializing scale factors using activation statistics, avoiding end-to-end distillation. This can be be enabled by adding the `--use_activation_aware` flag. Achieves 1.5-2x faster wall clock speed with lower perplexity.
