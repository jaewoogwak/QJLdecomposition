# Residual A–E ablation

All cases use the same packed exponent/mantissa Base.
Held-out routed activations are from a separate corpus from GPTQ calibration.

## Equal-budget comparison

- C_d2: residual bytes 3142287360; mean weight output error 0.15211918175013528; task accuracy 0.5047567541168223.
- D_d2: residual bytes 2995150056; mean weight output error 0.15452968141417606; task accuracy 0.5557480378299491.

Case D uses ResCode covariance initialization only, on the fixed mantissa residual.
Case E inference dequantizes packed INT2 before GEMM; its latency is not an optimized INT2 kernel.
