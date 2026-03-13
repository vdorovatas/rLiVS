print('started')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import torch.nn as nn
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
import sys
import re
from tqdm import tqdm
from utils import load_video_frames, split_into_short_clips_1_fps

warnings.filterwarnings("ignore")

def generate_questions(video_path,model,image_processor,tokenizer,device):
    captions=[]
    # Load and process video
    batch_size = len(video_path)
    image_tensors = []
    image_sizes = []
    first_layer_device = next(model.parameters()).device  # Get first parameter's device
    #input_data = input_data.to()
    
    all_frames, _, fps, _ = load_video_frames(video_path, 10)
    short_clips = split_into_short_clips_1_fps(all_frames=all_frames, fps=fps, frames_per_short_clip=10)
    for video_frames in short_clips:
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(first_layer_device)
        image_tensors.append(frames)
        image_size = [frame.size for frame in video_frames]
        image_sizes.append(image_size)

    # Prepare conversation input
    conv_template = "qwen_1_5"
    question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(first_layer_device)
    input_ids = input_ids.repeat(batch_size, 1).to(first_layer_device)
    modalities_list = ["video" for _ in range(batch_size)]
    # Generate response
    
    visual_tokens, cont = model.custom_generate(
        input_is_memory=False,
        inputs=input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=modalities_list,
        output_hidden_states=True
    )
    hidden_states_0 = torch.stack([c[:,-1,:] for c in cont.hidden_states[0]]).permute(1,0,2).unsqueeze(-3)
    hidden_states_rest = torch.stack([torch.stack(layer) for layer in cont.hidden_states[1:]]).squeeze(-2).permute(2,0,1,-1)
    latent_ouput = torch.cat((hidden_states_0, hidden_states_rest), dim=1)[:,:,-1, :]
    text_outputs = tokenizer.batch_decode(cont.sequences, skip_special_tokens=True)
    for t in text_outputs:
        captions.append(t)
    return captions, visual_tokens , latent_ouput 

    
# Load the OneVision model
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda:0"
device_map = 'auto' #"cuda:0"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

model.eval()

import os
import json

def save(data_list, output_file):
    for data in data_list:
        # Append the current video's data to the JSON file
        with open(output_file, 'r+', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_data.extend(data)
            f.seek(0)
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        

    print(f"Data batch successfully added to {output_file}")
    
def save_captions(data, output_file):
    with open(output_file, 'r+', encoding='utf-8') as f:
        existing_data = json.load(f)
        existing_data.extend(data)
        f.seek(0)
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"Data successfully saved to {output_file}")
    
def save_pth(data_list, output_file):
    torch.save(data_list, output_file)
    print(f"Visual Tokens batch successfully added to {output_file}")

def process_videos_in_folder(folder_path, tokens_file):
    # Iterate over all files in the folder
    video_metadata = {}
    save_idx = 0
    save_freq=1
    for file_number, filename in enumerate(tqdm(os.listdir(folder_path), desc="Processing files...", unit="file")):
        #data = []
        if filename.endswith(('.mp4', '.mkv', '.avi', '.mov')):  # Add video formats as needed
            video_path = os.path.join(folder_path, filename)
            if not os.path.exists(video_path): 
                print('ERROR: ', filename)
            filename = filename.rsplit('.', 1)[0]
            save_idx +=1

            captions, visual_tokens, latent_output = generate_questions(video_path,model,image_processor,tokenizer,device)
            video_metadata[filename] = {}
            for i in range(len(captions)):
                video_metadata[filename][i].append({
                    "visual_tokens": visual_tokens[i].detach().cpu(),
                    "latent_output": latent_output[i].detach().cpu(),
                    "caption": captions[i]
                })

            if save_idx % save_freq==0: 
                save_pth(video_metadata, tokens_file)

            del captions, visual_tokens, latent_output
    save_pth(video_metadata, tokens_file)
    print(f"Data successfully saved.")

folder_path = "/mnt/lustrefs/home/edo6236/streaming/datasets/next_qa/videos"
tokens_file = "xxx.pth"

process_videos_in_folder(folder_path, tokens_file)