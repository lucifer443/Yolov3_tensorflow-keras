TENSORFLOW_MODELS=$(cd "$(dirname $0)/.."; pwd)
#export PYTHONPATH=$TENSORFLOW_MODELS

MODEL_DIR=/home/changming/models/retinanet
python main.py \
  --strategy_type=mirrored \
  --num_gpus=4 \
  --model_dir="${MODEL_DIR}" \
  --mode=train \
  --config_file="my_retinanet.yaml"
