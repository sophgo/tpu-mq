python3 ptq-llm-OPTfp4.py --epochs 1 --b 4 --lr 1e-4 --wd 0.9 --wbit 4 --abit 8 --wob MinMaxObserver --aob MinMaxObserver --wfq FPXGROUPFakeQuantize  --afq E5M2FakeQuantize  2>&1 |tee a.txt
