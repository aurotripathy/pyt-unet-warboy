
Experiments with batch-size on a Unet-like model.

Since we're working with eandow weight and synthetic data, no accuracy measurements are made for now.


```
./convert_2_onnx.py --bs 2

./quantize.py --bs 2

./async_compile_run.py --bs 2
```



