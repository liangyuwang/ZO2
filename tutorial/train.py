
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from tutorial.configs import *
from tutorial import (
    GPT2ModelMezo,
    GPT2ModelMezoOffloading
)


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def check_peak_memory_usage(iter, device="cuda:0", use_tqdm=False):
    # Check the peak memory usage
    peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
    if use_tqdm:
        tqdm.write("Peak GPU Memory after iteration {}: {:.2f} MB".format(iter+1, peak_memory))
    else:
        print(f"Peak GPU Memory after iteration {iter+1}: {peak_memory:.2f} MB")
    torch.cuda.reset_peak_memory_stats(device=device)

def check_time_cost(iter, fn, *args, use_tqdm=False, **kwargs):
    t1 = time.time()
    out = fn(*args, **kwargs)
    t2 = time.time()
    time_cost = t2-t1
    throughtput = trainConfig.total_token_batch_size / time_cost
    if use_tqdm:
        tqdm.write("Time cost after iteration {}: {:.2f} ms, {:.2f} tok/s".format(iter+1, time_cost*1e3, throughtput))
    else:
        print("Time cost after iteration {}: {:.2f} ms, {:.2f} tok/s".format(iter+1, time_cost*1e3, throughtput))

def model_size(model: torch.nn.Module):
    total_size = sum(p.numel() for p in model.parameters())
    trainable_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total_size, "trainable": trainable_size}


def eval_acc():
    seed_everything(trainConfig.seed)
    mezoConfig.max_zo_random_seed = 1
    model_ref = GPT2ModelMezo(modelConfig, mezoConfig)
    total_parameters = model_size(model_ref)["total"]
    print(f"normal model size: {total_parameters/1024**3:.2f} B")
    offloadingConfig.offload_use_amp = False
    model = GPT2ModelMezoOffloading(modelConfig, mezoConfig, offloadingConfig)
    total_parameters = model_size(model)["total"]
    print(f"Offloading model size: {total_parameters/1024**3:.2f} B")
    for name_ref, p_ref in model_ref.named_parameters():
        for name, p in model.named_parameters():
            if name==name_ref:
                p_ref.data.copy_(p.data)
    model_ref = model_ref.to(modelConfig.device)

    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    
    # train
    for i in range(trainConfig.max_steps):
        model_ref.zo_training = True
        model.zo_training = True
        print(f"projected grad: {model_ref.projected_grad}, {model.projected_grad}")
        if offloadingConfig.offload_use_amp:
            with torch.autocast("cuda", offloadingConfig.offload_amp_dtype):
                out_ref = model_ref(**input)
                out = model(**input)
        else:
            out_ref = model_ref(**input)
            out = model(**input)
        (output1_ref, output2_ref), (loss1_ref, loss2_ref) = out_ref[:2]
        (output1, output2), (loss1, loss2) = out[:2]

    # eval
    model_ref.zo_training = False
    model.zo_training = False
    with torch.inference_mode():
        if offloadingConfig.offload_use_amp:
            with torch.autocast("cuda", offloadingConfig.offload_amp_dtype):
                eval_out_ref, eval_loss_ref = model_ref(**input)
                eval_out, eval_loss = model(**input)
        else:
            eval_out_ref, eval_loss_ref = model_ref(**input)
            eval_out, eval_loss = model(**input)

    print("loss1: ", loss1_ref.item(), loss1.item())
    print("loss2: ", loss2_ref.item(), loss2.item())
    print(torch.allclose(loss1_ref, loss1))
    print(torch.allclose(loss2_ref, loss2))
    print("diff loss1: ", torch.abs(loss1_ref - loss1).max())
    print("diff loss2: ", torch.abs(loss2_ref - loss2).max())
    print("diff output1: ", torch.abs(output1_ref - output1).max())
    print("diff output2: ", torch.abs(output2_ref - output2).max())
    print("diff projected_grad: ", torch.abs(model_ref.projected_grad - model.projected_grad).max())
    print("diff eval loss: ", torch.abs(eval_loss_ref - eval_loss).max())
    print("diff eval output: ", torch.abs(eval_out_ref - eval_out).max())

def mezo_performance():
    seed_everything(trainConfig.seed)
    model_ref = GPT2ModelMezo(modelConfig, mezoConfig)
    total_parameters = model_size(model_ref)["total"]
    print(f"normal model size: {total_parameters/1024**3:.2f} B")
    model_ref = model_ref.to(modelConfig.device)
    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    torch.cuda.reset_peak_memory_stats()
    for i in tqdm(range(trainConfig.max_steps)):
        if offloadingConfig.offload_use_amp:
            with torch.autocast("cuda", offloadingConfig.offload_amp_dtype):
                check_time_cost(i, model_ref, **input)
                check_peak_memory_usage(i, modelConfig.device, True)
        else:
            check_time_cost(i, model_ref, **input)
            check_peak_memory_usage(i, modelConfig.device, True)

def mezo_offloading_performance(overlap=True):
    seed_everything(trainConfig.seed)
    model = GPT2ModelMezoOffloading(modelConfig, mezoConfig, offloadingConfig)
    model.overlap = overlap
    total_parameters = model_size(model)["total"]
    print(f"Offloading model size: {total_parameters/1024**3:.2f} B")
    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    torch.cuda.reset_peak_memory_stats()
    for i in tqdm(range(trainConfig.max_steps)):
        if offloadingConfig.offload_use_amp:
            with torch.autocast("cuda", offloadingConfig.offload_amp_dtype):
                check_time_cost(i, model, **input)
                check_peak_memory_usage(i, modelConfig.device, True)
        else:
            check_time_cost(i, model, **input)
            check_peak_memory_usage(i, modelConfig.device, True)

def train_torch():
    seed_everything(trainConfig.seed)
    model_ref = GPT2ModelMezo(modelConfig, mezoConfig)
    total_parameters = model_size(model_ref)["total"]
    print(f"normal model size: {total_parameters/1024**3:.2f} B")
    model_ref = model_ref.to(modelConfig.device)
    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    opt = torch.optim.AdamW(model_ref.parameters())
    for i in tqdm(range(trainConfig.max_steps)):
        # train
        model_ref.zo_training = False
        model_ref.grad_accum = False
        if offloadingConfig.offload_use_amp:
            with torch.autocast("cuda", offloadingConfig.offload_amp_dtype):
                _, loss = model_ref(**input)
        else:
            _, loss = model_ref(**input)
        loss.backward()
        opt.step()
        tqdm.write("Iteration {}, loss: {}".format(i, loss))

def train_mezo():
    seed_everything(trainConfig.seed)
    model_ref = GPT2ModelMezo(modelConfig, mezoConfig)
    total_parameters = model_size(model_ref)["total"]
    print(f"normal model size: {total_parameters/1024**3:.2f} B")
    model_ref = model_ref.to(modelConfig.device)
    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    for i in tqdm(range(trainConfig.max_steps)):
        # train
        model_ref.zo_training = True
        model_ref.grad_accum = False
        if offloadingConfig.offload_use_amp:
            with torch.autocast("cuda", offloadingConfig.offload_amp_dtype):
                (_, _), (loss, _) = model_ref(**input)
        else:
            (_, _), (loss, _) = model_ref(**input)
        tqdm.write("Iteration {}, loss: {}, projected grad: {}".format(i, loss, model_ref.projected_grad.abs()))

def train_mezo_offloading():
    seed_everything(trainConfig.seed)
    model = GPT2ModelMezoOffloading(modelConfig, mezoConfig, offloadingConfig)
    total_parameters = model_size(model)["total"]
    print(f"normal model size: {total_parameters/1024**3:.2f} B")
    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    for i in tqdm(range(trainConfig.max_steps)):
        # train
        model.zo_training = True
        model.grad_accum = False
        if offloadingConfig.offload_use_amp:
            with torch.autocast("cuda", offloadingConfig.offload_amp_dtype):
                (_, _), (loss, _) = model(**input)
        else:
            (_, _), (loss, _) = model(**input)
        tqdm.write("Iteration {}, loss: {}, projected grad: {}".format(i, loss, model.projected_grad.abs()))

def eval_mezo():
    seed_everything(trainConfig.seed)
    model_ref = GPT2ModelMezo(modelConfig, mezoConfig)
    total_parameters = model_size(model_ref)["total"]
    print(f"normal model size: {total_parameters/1024**3:.2f} B")
    model_ref = model_ref.to(modelConfig.device)
    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    model_ref.zo_training = False
    model_ref.training = False
    with torch.inference_mode():
        _, loss = model_ref(**input)
        print(f"loss: {loss}")

def eval_mezo_offloading():
    seed_everything(trainConfig.seed)
    model = GPT2ModelMezoOffloading(modelConfig, mezoConfig, offloadingConfig)
    total_parameters = model_size(model)["total"]
    print(f"normal model size: {total_parameters/1024**3:.2f} B")
    print("Init dataset")
    B, T = modelConfig.batch_size, modelConfig.block_size
    data = torch.randint(0, modelConfig.vocab_size, (B, T+1)).to(modelConfig.device)
    x = data[:, 0:T].clone()
    y = data[:, 1:T+1].clone()
    input = {"idx": x, "targets": y}
    model.zo_training = False
    model.grad_accum = False
    with torch.inference_mode():
        _, loss = model(**input)
        print(f"loss: {loss}")

if __name__=="__main__":
    modelConfig = OPT_125m()
    trainConfig = TrainConfig()
    mezoConfig = MezoConfig()
    offloadingConfig = OffloadingConfig()

    # eval_acc()
    # mezo_performance()
    mezo_offloading_performance(overlap=True)

    # train_torch()
    # train_mezo()
    # train_mezo_offloading()

    # eval_mezo()
    # eval_mezo_offloading()