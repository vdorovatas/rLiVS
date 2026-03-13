from torch.utils.data import Dataset
import json
from typing import Any
import glob
import torch
import copy
import os
import re
from utils import MemoryTreeNode
import math

class MovieChatEval(Dataset):
    # Initialize class
    def __init__(self, args, **kwargs: Any) -> None:
        self.args = args
        self.dataset=glob.glob(args.jsons_path+'*')
    # Overload getitem function
    def __getitem__(self, index):
        contents = read_json(self.dataset[index])
        video_name=contents['info']['video_path'].split('.')[0]
        memory_tokens_path=os.path.join(self.args.memories_folder,'memories_{}.pth'.format(video_name))
        memory_tokens=torch.load(memory_tokens_path, weights_only=False, map_location="cpu")
        caption=copy.deepcopy(contents['caption'])
        fps=copy.deepcopy(contents['info']['fps'])
        # Extracting questions and answers separately
        glob_questions = [item["question"] for item in contents['global']]
        glob_answers = [item["answer"] for item in contents['global']]
        assert len(glob_questions) == len(glob_answers)
        # Extracting questions and answers separately
        bp_timestamps=[item["time"] for item in contents['breakpoint']]
        bp_questions = [item["question"] for item in contents['breakpoint']]
        bp_answers = [item["answer"] for item in contents['breakpoint']]
        return video_name, memory_tokens, caption, glob_questions, glob_answers, bp_timestamps, bp_questions, bp_answers,fps
    # Overload len function
    def __len__(self):
        return len(self.dataset)


class StreamBenchEval(Dataset):
    # Initialize class
    def __init__(self, args, **kwargs: Any) -> None:
        self.args = args
        self.dataset=args.jsons_path
        self.contents = read_json(self.dataset)
        #self.video_folder="/shared/home/EDO6236/streaming/datasets/StreamBench/videos/"
    # Overload getitem function
    def __getitem__(self, index):
        contents=self.contents[index]
        #print(contents)
        #print('###########')
        '''
        vr = VideoReader(os.join(self.video_folder, contents['info']['video_path']), ctx=cpu(0))
        actual_fps = vr.get_avg_fps()
        total_frame_num = len(vr)
        frame_index = int(timestamp * actual_fps)
        '''
        video_name=contents['info']['video_path'].split('.')[0]
        memory_tokens_path=os.path.join(self.args.memories_folder,'memories_{}.pth'.format(video_name))
        if not os.path.exists(memory_tokens_path):
            print(f'Error in video {video_name}')
            return None, None, None, None, None
        memory_tokens=torch.load(memory_tokens_path, weights_only=False, map_location="cpu")
        #caption=copy.deepcopy(contents['caption'])
        #fps=copy.deepcopy(contents['info']['fps'])
        # Extracting questions and answers separately
        questions = [item["question"] for item in contents['breakpoint']]
        answers = [item["answer"] for item in contents['breakpoint']]
        timestamps = [item["time"] for item in contents['breakpoint']]
        assert len(questions) == len(answers)        
        return video_name, memory_tokens, questions, answers, timestamps
    # Overload len function
    def __len__(self):
        return len(self.contents)
class MovieNetEval(Dataset):
    # Initialize class
    def __init__(self, args, **kwargs: Any) -> None:
        self.args = args
        self.dataset=glob.glob(args.jsons_path+'*')
    # Overload getitem function
    def __getitem__(self, index):
        contents = read_json(self.dataset[index])
        video_name=contents['breakpoint'][0]['video_id']
        memory_tokens_path=os.path.join(self.args.memories_folder,'memories_{}.pth'.format(video_name))
        memory_tokens=torch.load(memory_tokens_path, weights_only=False, map_location="cpu")
        # Extracting questions and answers separately
        questions = [item["question"] for item in contents['breakpoint']]
        answers = [item["answer"] for item in contents['breakpoint']]
        assert len(questions) == len(answers)
        if self.args.dataset=='MovieNet-stream':
            vid_start_time,vid_end_time = get_sorted_frames_and_shot_range(os.path.join(self.args.frames_folder,video_name))
        else:
            vid_start_time,vid_end_time = find_first_and_last_image_numbers(os.path.join(self.args.frames_folder,video_name))
        if self.args.dataset=='MovieNet-stream':
            start_times=[int(item["start_time"].split('_')[1]) for item in contents['breakpoint']]
        else:
            start_times=[item["start_time"] for item in contents['breakpoint']]
        end_times=[]
        #gt_end_times=[item["end_time"] for item in contents['breakpoint']]
        for item in contents['breakpoint']:
            try:
                if self.args.dataset=='MovieNet-stream':
                    end_times.append(int(item["end_time"].split('_')[1]))
                else:
                    end_times.append(item["end_time"])
            except:
                if self.args.dataset=='MovieNet-stream':
                    end_times.append(int(item["start_time"].split('_')[1]))
                else:
                    end_times.append(item["start_time"])
        return video_name, memory_tokens, questions, answers, start_times, end_times, vid_start_time,vid_end_time
    # Overload len function
    def __len__(self):
        return len(self.dataset)

class MovieNetEval_video(Dataset):
    # Initialize class
    def __init__(self, args, **kwargs: Any) -> None:
        self.args = args
        self.dataset=glob.glob(args.jsons_path+'*')
    # Overload getitem function
    def __getitem__(self, index):
        contents = read_json(self.dataset[index])
        video_name=contents['global'][0]['video_id']
        memory_tokens_path=os.path.join(self.args.memories_folder,'memories_{}.pth'.format(video_name))
        memory_tokens=torch.load(memory_tokens_path, weights_only=False, map_location="cpu")
        # Extracting questions and answers separately
        questions = [item["question"] for item in contents['global']]
        answers = [item["answer"] for item in contents['global']]
        assert len(questions) == len(answers)
        #start_times=[item["start_time"] for item in contents['global']]
        #end_times=[]
        return video_name, memory_tokens, questions, answers, None, None
    # Overload len function
    def __len__(self):
        return len(self.dataset)
    
def read_json(file_path):
    #print(file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data as a Python dictionary
    return data
# Overload collate function. Parse batch and remove data that are None due to data filtering
def _collate_fn(batch):
    collate = None
    is_data = any([True for b in batch if b is not None])
    if is_data:
        batch = list(filter(lambda x: x is not None, batch))
        # collate = default_collate(batch)
        collate = list(map(list, zip(*batch)))
    return collate
def get_eval_loader(args):
    if args.dataset=='CG-Bench':
        my_dataset=CGBenchEval(args)
    elif args.dataset=='StreamingBench':
        my_dataset=StreamingBenchEval(args)
    elif args.dataset=='StreamBench':
        my_dataset=StreamBenchEval(args)
    elif args.dataset=='MovieChat':
        my_dataset=MovieChatEval(args)
    elif args.dataset=='MovieNet-stream' or args.dataset=='Ego4d-stream':
        my_dataset=MovieNetEval(args)
    elif args.dataset=='MovieNet-global' or args.dataset=='Ego4d-global':
        my_dataset=MovieNetEval_video(args)
    data_loader=torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size, shuffle=args.shuffle,
                                                    num_workers=args.num_workers, pin_memory=False, drop_last=True, collate_fn=_collate_fn)
    return data_loader


def get_sorted_frames_and_shot_range(folder_path):
    # Regex pattern for matching and extracting shot and image indices
    pattern = re.compile(r"shot_(\d+)_img_(\d+)\.jpg")

    # List all matching files
    matched_files = []
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            shot_index = int(match.group(1))
            img_index = int(match.group(2))
            matched_files.append((shot_index, img_index, filename))

    # Sort based on shot index first, then image index
    matched_files.sort()

    # Extract sorted filenames
    sorted_filenames = [filename for _, _, filename in matched_files]

    # Determine the range of shot indices
    if matched_files:
        start_shot = matched_files[0][0]
        end_shot = matched_files[-1][0]
    else:
        start_shot = end_shot = None

    return start_shot, end_shot

def find_first_and_last_image_numbers(folder_path):     
    pattern = re.compile(r'^(\d+)\.jpg$', re.IGNORECASE)
    
    numbers = []
    
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            numbers.append(number)
    
    if not numbers:
        print("No matching .jpg files found.")
        return None, None
    
    numbers.sort()
    first = numbers[0]
    last = numbers[-1]
    
    return first, last  

class StreamingBenchEval(Dataset):
    # Initialize class
    def __init__(self, args, **kwargs: Any) -> None:
        self.args = args
        with open(args.jsons_path, "r") as f:
            self.contents = json.load(f)
        self.video_names = list(self.contents.keys())

    def __getitem__(self, index):
        contents = self.contents[self.video_names[index]] 
        video_name=self.video_names[index]
        memory_tokens_path=os.path.join(self.args.memories_folder,'memories_{}.pth'.format(video_name))
        if not os.path.exists(memory_tokens_path):
            print(f'Error in video {video_name}')
            return None, None, None, None, None, None
        memory_tokens=torch.load(memory_tokens_path, weights_only=False, map_location="cpu")
        questions = contents['question']
        answers = contents['answer']
        choices = contents['options']
        timestamps = contents['timestamp']
        timestamps = [math.ceil(self.timestamp_to_seconds(ts) / 16) for ts in timestamps]
        assert len(questions) == len(answers)
        return video_name, memory_tokens, questions, answers, choices, timestamps
    # Overload len function
    def __len__(self):
        return len(self.video_names)

    def timestamp_to_seconds(self, ts):
        parts = list(map(int, ts.split(":")))
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        else:
            raise ValueError(f"Invalid timestamp format: {ts}")
        return h * 3600 + m * 60 + s

import string
class CGBenchEval(Dataset):
    # Initialize class
    def __init__(self, args, **kwargs: Any) -> None:
        self.args = args
        with open(args.jsons_path, "r") as f:
            self.contents = json.load(f)
        self.video_names = list(self.contents.keys())

    def __getitem__(self, index):
        contents = self.contents[self.video_names[index]]
        video_name=self.video_names[index]
        memory_tokens_path=os.path.join(self.args.memories_folder,'memories_{}.pth'.format(video_name))
        if not os.path.exists(memory_tokens_path):
            print(f'Error in video {video_name}')
            return None, None, None, None, None
        memory_tokens=torch.load(memory_tokens_path, weights_only=False, map_location="cpu")
        questions = contents['questions']
        choices, answers = [], []
        for cs,a in zip(contents['choices'], contents['answers']):
            f_cs, f_a = self.format_choices_and_answer(cs,a)
            choices.append(f_cs)
            answers.append(f_a)
        assert len(questions) == len(answers)
        return video_name, memory_tokens, questions, answers, choices
    # Overload len function
    def __len__(self):
        return len(self.video_names)

    def format_choices_and_answer(self, choices, answer):
        letters = string.ascii_uppercase
        formatted_choices = [f"{letter}. \"{choice}\"" for letter, choice in zip(letters, choices)]
        answer_idx = choices.index(answer)
        answer_letter = letters[answer_idx]
        return formatted_choices, answer_letter



