cd /hy-tmp/src
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp79.json -gpu
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp80.json -gpu
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp81.json -gpu
python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py /hy-tmp/src/configs/exp82.json -gpu

cd /hy-tmp/results
zip exp79-82data.zip ./*.txt
rm ./*.txt
oss cp exp79-82data.zip oss://exp79-82data.zip

cd /hy-tmp/temp
zip exp79-82.zip ./*.pth
rm ./*.pth
oss cp exp79-82.zip oss://exp79-82.zip

shutdown