# Run from Backprojection_Loss directory
# Dataset is located in ../DATASET/ relative to this script
# Use --no_cuda flag to run on CPU (required for Mac without GPU)
python main.py \
  --no_cuda \
  --image_dir ../DATASET/images \
  --gt_dir ../DATASET/ground_truth \
  --num_train 2535 \
  --loss_policy backproject \
  --save_freq 100 \
  --weight_init xavier \
  --use_cholesky 0 \
  --split_percentage 0.1 \
  --activation_layer square \
  --pretrained false \
  --pretrain_epochs 25 \
  --skip_epochs 25 \
  --nclasses 4 \
  --mask_percentage 0.20 \
  --order 3 \
  --clas 1 \
  --nepochs 400


