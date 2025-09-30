#!/bin/bash
# PTQ test
python ./tpu_mq/ptq_train_all_model.py

# QAT test
python ./tpu_mq/qat_train_all_model.py