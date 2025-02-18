nohup python -u main.py -t 1 -jr 1 -gr 40 -nc 20 -nb 100 -data Cifar100dir0.1 -m cnn -lr 0.005 -algo FLAYER -p 2 -did 0 > cifar100_cnn_flayer.out 2>&1 &

#nohup python -u main.py -t 1 -jr 1 -gr 75 -nc 20 -nb 100 -data Cifar100dir0.1 -m resnet -lr 0.1 -algo FLAYER -p 4 -did 0 > cifar100_resnet_flayer.out 2>&1 &

#nohup python -u main.py -t 1 -jr 1 -gr 100 -nc 20 -nb 10 -data Cifar10 -m cnn -lr 0.005 -algo FLAYER -p 2 -did 0 > cifar10_cnn_flayer.out 2>&1 &

#nohup python -u main.py -t 1 -jr 1 -gr 100 -nc 20 -nb 10 -data Cifar10 -m resnet -lr 0.1 -algo FLAYER -p 4 -did 0 > cifar10_resnet_flayer.out 2>&1 &

#nohup python -u main.py -t 1 -jr 1 -gr 25 -nc 20 -nb 200 -data Tiny-imagenet -m cnn -lr 0.005 -algo FLAYER -p 2 -did 0 > tiny_cnn_flayer.out 2>&1 &

#nohup python -u main.py -t 1 -jr 1 -gr 25 -nc 20 -nb 200 -data Tiny-imagenet -m resnet -lr 0.1 -algo FLAYER -p 4 -did 0 > tiny_resnet_flayer.out 2>&1 &

#nohup python -u main.py -t 1 -jr 1 -gr 1000 -nc 20 -nb 4 -data agnews -m fastText -lr 0.1 -algo FLAYER -p 2 -did 0 > agnews_fasttext_flayer.out 2>&1 &



