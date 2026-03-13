import os
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from model import rLiVS
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
import sys
from utils import load_video_frames, split_into_short_clips_1_fps, get_all_memories_in_order, frames_from_images_path
import glob
from tqdm import tqdm
import sys
import argparse

def setup_args():
    parser = argparse.ArgumentParser('Argparse options for feature extraction pipeline')
    parser.add_argument('--dataset', default='movienet', help='Dataset to evaluate [movienet, ego4d].')
    args, _ = parser.parse_known_args()
    return args

args=setup_args()

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")
        
def chunk_list(lst, size):
    """Splits lst into chunks of given size."""
    return [lst[i:i+size] for i in range(0, len(lst), size)]

if args.dataset == 'movienet':
    video_folder="/fsx/ad/project-vaggelis/VStream-QA/vstream-realtime/movienet_frames/"
    save_path="llava_ov_memories/rs_movienet"
elif args.dataset == 'ego4d':
    video_folder="..."
    save_path="..."
else:
    print('dataset error')
    sys.exit()

warnings.filterwarnings("ignore")
pretrained= "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}

tokenizer, vlm, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="float16", device_map=device_map, attn_implementation="flash_attention_2", **llava_model_args)

vlm.eval()

device = "cuda:0"

model = rLiVS(vlm, device, tokenizer)


create_directory(save_path)
video_paths=glob.glob(video_folder+"*")
## hyperparams
RAW_CONTEXT_SIZE=16
SELECT=1
SELECTION = 196
short_term_mem_size = 16
#############
#for i, path in enumerate(tqdm(video_paths, desc="Processing videos", unit="video")):
for i, path in enumerate(tqdm(video_paths, desc="Processing videos", unit="video")):  # processes from end
    vid_name=path.split('/')[-1].split('.')[0]
    ####  check if it already exists 
    file_path = "{}/memories_{}.pth".format(save_path, vid_name)
    
    if os.path.exists(file_path):
        print(f"File {vid_name} already exists, skipping...")
        continue                                        
    ####   
    image_tensors, image_sizes = frames_from_images_path(device, path, image_processor, RAW_CONTEXT_SIZE, 1)
    with torch.no_grad():
        memory = model.recurrent_simple(image_tensors, image_sizes, short_term_mem_size, SELECTION)
        torch.save(memory, "{}/memories_{}.pth".format(save_path,vid_name))
        torch.cuda.empty_cache()

