TRAIN_CONFIG="configs/pf-lam-20k.yaml"
MODEL_NAME="test_data/weights/pf-lam-20k_step-90k/"
IMAGE_INPUT="test_data/inference_imgs/093_EMO-2/"
GS_INPUT="test_data/inference_imgs/093_EMO-2/"
MOTION_SEQS_DIR="test_data/motion_seqs/093_EMO-2/"  # MUST END WITH '/'
IMAGE_INPUT_NAME="1.png"


TRAIN_CONFIG=${1:-$TRAIN_CONFIG}
MODEL_NAME=${2:-$MODEL_NAME}
IMAGE_INPUT=${3:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${4:-$MOTION_SEQS_DIR}
IMAGE_INPUT_NAME=${5:-$IMAGE_INPUT_NAME}
GS_INPUT=${6:-$GS_INPUT}

echo "TRAIN_CONFIG: $TRAIN_CONFIG"
echo "MODEL_NAME: $MODEL_NAME"
echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR" 
echo "IMAGE_INPUT_NAME: $IMAGE_INPUT_NAME"
echo "GS_INPUT: $GS_INPUT"

export PYTHONPATH=$PYTHONPATH:$pwd
python -m gstransformer.launch infer.lam --config $TRAIN_CONFIG \
        model_name=$MODEL_NAME \
        image_input=$IMAGE_INPUT \
        motion_seqs_dir=$MOTION_SEQS_DIR \
        input_image_name=$IMAGE_INPUT_NAME \
        gs_input=$GS_INPUT \
        motion_img_dir=null  \
        motion_img_need_mask=true \
        cross_id=true \
        rank=0 \
        nodes=0 \
        render_fps=20 \
        motion_video_read_fps=30 \
        export_video=true \
        export_mesh=false \
        save_ply=false \
        save_img=false \
        save_img_type='cat' \
        test_sample=false \
        cycle_sample_mv_cam=false \
        frames_per_loop=128 \
        vis_len=null \
        vis_motion=true \
        vis_rgb=true \
        vis_blend_res=false \
        vis_ref=true \
        