eval "$(conda shell.bash hook)"
conda activate DL4NLP_yolo

#REMEMBER TO PUT NUMBER IF CALLS TO 20!!! Bayesian Optimization

python3 yolo/yolo_whisper.py --model nano --image-size 640
python3 yolo/yolo_whisper.py --model medium --image-size 640
python3 yolo/yolo_whisper.py --model nano --image-size 1216