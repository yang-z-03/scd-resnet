cd /hy-tmp/src
# python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp83.json -gpu
# python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp84.json -gpu
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp85.json -gpu
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp86.json -gpu

cd /hy-tmp/results
zip exp85-86data.zip ./*.txt

oss cp exp85-86data.zip oss://exp85-86data.zip

cd /hy-tmp/temp
zip exp85-86.zip ./*.pth

oss cp exp85-86.zip oss://exp85-86.zip

shutdown