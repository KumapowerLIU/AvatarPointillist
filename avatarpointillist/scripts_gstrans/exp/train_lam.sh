ACC_CONFIG="./configs/accelerate-train.yaml"

if [ -n "$1" ]; then
  TRAIN_CONFIG=$1
else
  TRAIN_CONFIG="./configs/lam-20k-hongyu-aug.yaml"
fi

accelerate launch  --main_process_port 29508 --config_file $ACC_CONFIG -m gstransformer.launch train.lam --config $TRAIN_CONFIG