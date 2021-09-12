# EndoCV2021_yolov5
# The third place solution for detection part of the 3rd International Endoscopy Computer Vision Challenge and Workshop  (EndoCV2021)

To install the required environment, first run:
`pip install -r requirements.txt`

Then, train the yolov5x-p6 model for polyp detection:
(note: we use 2 NVIDIA Tesla V100 32GB GPU for training and one for testing, the yolov5x6.pt file can be downloaded in the website of yolov5:https://github.com/ultralytics/yolov5)
`python -u -m torch.distributed.launch --nproc_per_node 2 train.py --name [your expriment name] --img 1280 --batch-size 8 --epochs 300 --data trainData_EndoCV2021_yolo_9_1_V2.yaml --multi-scale --weights yolov5x6.pt --cfg yolov5x6.yaml --device 0,1 --hyp hyp.finetune.yaml --sync-bn`

For inference, run:
`python -u endocv2021_test.py --weights [your best .pt checkpoint files] --source [test data] --img-size 1536 --augment --name [your expriment name] --iou-thres 0.6 --save-txt --save-conf`
