
# system requirements

the version status of the training environment is list below:
* conda 4.8.2
* pytorch 1.7.1
* torchvision 0.8.2
* python 3.8, 3.8.12

# training

```bash
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /configs/center.json -gpu
```

# preprocess

```bash
python preprocess.py -i "<path_to_dataset>/Images/" -a "<path_to_dataset>/Annotations/" -s 512 -t 0.5 -m "0 0 0 0" -p "datasets.scds.scdx16p100" "<path_to_the_result_dataset_file>.d"
```