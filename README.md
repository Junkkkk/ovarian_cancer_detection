# ovarian cancer detection
Prediction ovarian cancer regions with whole slide images

# Train code
$ python train.py -h
usage: train.py [-h] [--folder FOLDER] [--data_path DATA_PATH] [--save_path SAVE_PATH] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--epochs EPOCHS] [--gpu_num GPU_NUM]
                [--model_num MODEL_NUM]

optional arguments:
  -h, --help            show this help message and exit
  --folder FOLDER       (default=0)
  --data_path DATA_PATH
                        (default=./data)
  --save_path SAVE_PATH
                        (default=./result)
  --batch_size BATCH_SIZE
                        (default=16)
  --num_workers NUM_WORKERS
                        (default=8)
  --epochs EPOCHS       (default=150)
  --gpu_num GPU_NUM     (default="0,1")
  --model_num MODEL_NUM (default=0) $
