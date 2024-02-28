# Benchmark for instantiating `onnxruntime.InferenceSession`

There has been a large regression in the session creation time between onnxruntime 1.16.3 and 1.17.0. Running `bench.py` with `onnxruntime=1.17.0` on `osx-arm64` yields:

```shell
$ python bench.py 

onnx.__version__='1.15.0'
ort.__version__='1.17.0'
n_parallel_subgraphs=1000
n_stages=1

Constructing: 0.752s
Building: 1.44s
Session creation: 28.8s
```

while `onnxruntime=1.16.3` exhibits much better performance

```shell
$ python models.py 

onnx.__version__='1.15.0'
ort.__version__='1.16.3'
n_parallel_subgraphs=1000
n_stages=1

Constructing: 0.753s
Building: 1.47s
Session creation: 1.0s
```
