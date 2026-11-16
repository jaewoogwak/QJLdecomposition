# Residual A–E ablation

All cases use the same packed exponent/mantissa Base.
Held-out routed activations are from a separate corpus from GPTQ calibration.

## Equal-budget comparison


Case D uses ResCode covariance initialization only, on the fixed mantissa residual.
Case E inference dequantizes packed INT2 before GEMM; its latency is not an optimized INT2 kernel.
No downstream accuracy is reported until evaluation completes.
