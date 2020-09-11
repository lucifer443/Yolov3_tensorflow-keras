export PYTHONPATH=${HOME}/codebase/inference_quant/faster_rcnn_resnet101
cd ${HOME}/codebase/tf-raw/models/research/object_detection/dataset_tools
RAW_DIR=/algo_public/datasets/COCO17
TRAIN_IMAGE_DIR=${RAW_DIR}/train2017
VAL_IMAGE_DIR=${RAW_DIR}/val2017
TEST_IMAGE_DIR=${RAW_DIR}/test2017
TRAIN_ANNOTATIONS_FILE=${RAW_DIR}/annotations/instances_train2017.json
VAL_ANNOTATIONS_FILE=${RAW_DIR}/annotations/instances_val2017.json
TESTDEV_ANNOTATIONS_FILE=${RAW_DIR}/annotations/image_info_test-dev2017.json
OUTPUT_DIR=/algo_proj/changming/model-zoo/tfrecords/
python create_coco_tf_record.py --logtostderr \
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --test_image_dir="${TEST_IMAGE_DIR}" \
  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
  --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}" \
  --include_maskes
