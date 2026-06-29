#!/bin/bash
exec python3 -u train.py --n-per-class 500 --epochs 50 --batch-size 48 --lr 8e-4 --patience 10 --seed 2024
