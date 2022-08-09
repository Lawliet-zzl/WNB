DATASET='CIFAR10' # ('CIFAR10', 'CIFAR100' 'SVHN')
MODEL='resnet' # ('resnet' 'senet' 'shufflenet')
N='100' # (10 100 1000)
C=0.001 # (0.01 0.001 0.0001)
NAME='0'

python main.py --model ${MODEL} --dataset ${DATASET} --name=${NAME} --C ${C} --N ${N}\
