# MeZO-offloading

## Emvironment
* PyTorch 2.4.0 (CUDA 12.1)

## Getting Started
In tutorial/train.py, there's some line:
* You can change "from tutorial.nanogpt_mezo_offloading_v2 import GPT2ModelMezoOffloading", with v1 or v2.
* You can change the configs:
```python
    modelConfig = OPT_125m()
    trainConfig = TrainConfig()
    mezoConfig = MezoConfig()
    offloadingConfig = OffloadingConfig()
```
* You can try different test by uncommenting it:
```python
    eval_acc()
    # mezo_performance()
    # mezo_offloading_performance(overlap=True)

    # train_torch()
    # train_mezo()
    # train_mezo_offloading()

    # eval_mezo()
    # eval_mezo_offloading()
```

After that, you can run the following commands to run.
```shell
python tutorial/train.py
```
You can also run the following code to monitor both GPU and CPU memory, 
```shell
bash execute.sh
```
or use nsys profiler to optim throughtput.
```shell
nsys profile --stats=true --trace=nvtx,cuda python tutorial/train.py
```

## Feature
* Overlap
    * args: overlap
* Offload only blocks
    * args: offload_every_blocks
* AMP