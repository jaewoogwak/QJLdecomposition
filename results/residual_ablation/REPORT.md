# Residual A–E ablation

All cases use the same packed exponent/mantissa Base.
Held-out routed activations are from a separate corpus from GPTQ calibration.

## Equal-budget comparison

- C_d2: residual bytes 3142287360; mean weight output error 0.15202550264765266; task accuracy None.
- D_d2: residual bytes 2995150056; mean weight output error 0.15446791580826458; task accuracy None.
- E: residual bytes 3142287360; mean weight output error 0.20024446121296258; task accuracy None.

Case D uses ResCode covariance initialization only, on the fixed mantissa residual.
Case E inference dequantizes packed INT2 before GEMM; its latency is not an optimized INT2 kernel.
No downstream accuracy is reported until evaluation completes.
