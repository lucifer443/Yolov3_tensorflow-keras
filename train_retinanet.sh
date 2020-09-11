TENSORFLOW_MODELS=$(cd "$(dirname $0)/.."; pwd)
#export PYTHONPATH=$TENSORFLOW_MODELS
export CUDA_VISIBLE_DEVICES=0

MODEL_DIR=/algo_proj/changming/model-zoo/retinanet
python main.py \
  --strategy_type=mirrored \
  --num_gpus=4 \
  --model_dir="${MODEL_DIR}" \
  --mode=train \
  --config_file="my_retinanet.yaml"
