
Experiments with batch-size on a Unet-like model.

Since we're working with random weights and synthetic data, no accuracy measurements can be made, only perfformance measurements can be made.


```
./convert_2_onnx.py --bs 2

./quantize.py --bs 2

./async_compile_run.py --bs 2
```



