cd /hy-tmp/src
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp74.json -gpu
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp75.json -gpu
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp76.json -gpu
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp77.json -gpu
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp78.json -gpu

cd /hy-tmp/results
zip exp74-78data.zip ./*.txt
rm ./*.txt
oss cp exp74-78data.zip oss://exp74-78data.zip

cd /hy-tmp/temp
zip exp74-78.zip ./*.pth
rm ./*.pth
oss cp exp74-78.zip oss://exp74-78.zip

shutdown