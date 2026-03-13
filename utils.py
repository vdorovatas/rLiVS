
import numpy as np
from decord import VideoReader, cpu
import torch
import torch.nn.functional as F
import os
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import random
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset

def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    spare_frames = spare_frames.astype(np.float32)
    return spare_frames  # (frames, height, width, channels)

# Function to extract frames from video
def load_video_frames(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    frame_idx = list(range(total_frame_num))
    all_frames = vr.get_batch(frame_idx).asnumpy()
    all_frames = all_frames.astype(np.float32)
    fps = vr.get_avg_fps()
    duration = total_frame_num / fps
    return all_frames, total_frame_num ,fps, duration  # (frames, height, width, channels)

def split_into_short_clips_1_fps(all_frames, fps, frames_per_short_clip):
    sample_every = round(fps)
    short_clip_frames = []
    short_clips_frames_list = []

    for i in range(0, len(all_frames), sample_every):
        
        # uniformly sample from the sample_every frames subset
        start = i
        end = min(i + sample_every, len(all_frames))
        sampled_frame = random.choice(all_frames[start:end])
        short_clip_frames.append(sampled_frame)

        if len(short_clip_frames) == frames_per_short_clip:
            short_clips_frames_list.append(short_clip_frames)
            short_clip_frames = []

    # Add any remaining frames to the result
    if short_clip_frames:
        short_clips_frames_list.append(short_clip_frames)

    return short_clips_frames_list


def prepare_input(question, answer, tokenizer):
    tokenizer.padding_side = "left"
    conv_template = "qwen_1_5"
    instruction = f"{DEFAULT_IMAGE_TOKEN}\n" + question + "."

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    full_input_text = prompt_question + answer + tokenizer.eos_token
    tokenized = tokenizer_image_token(full_input_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = tokenized[:,:-1]
    target_ids = tokenized[:, 1:]
    tokenized_labels = tokenizer(answer+tokenizer.eos_token, return_tensors='pt', padding='max_length', max_length=target_ids.shape[1])["input_ids"]
    label_ids_length = (tokenized_labels != tokenizer.pad_token_id).sum().item()
    tokenized_labels = torch.where(tokenized_labels != tokenizer.pad_token_id, tokenized_labels, -100)

    return input_ids, tokenized_labels, label_ids_length

def prepare_input_multiple_choice(question, answer, tokenizer):
    tokenizer.padding_side = "left"
    conv_template = "qwen_1_5"
    instruction = f"{DEFAULT_IMAGE_TOKEN}\n" + question

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    full_input_text = prompt_question + answer + tokenizer.eos_token
    tokenized = tokenizer_image_token(full_input_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    input_ids = tokenized[:,:-1]
    target_ids = tokenized[:, 1:]
    tokenized_labels = tokenizer(answer+tokenizer.eos_token, return_tensors='pt', padding='max_length', max_length=target_ids.shape[1])["input_ids"]
    label_ids_length = (tokenized_labels != tokenizer.pad_token_id).sum().item()
    tokenized_labels = torch.where(tokenized_labels != tokenizer.pad_token_id, tokenized_labels, -100)

    return input_ids, tokenized_labels, label_ids_length

def prepare_input_multiple_choice_generation(question, tokenizer):
    tokenizer.padding_side = "left"
    conv_template = "qwen_1_5"
    instruction = f"{DEFAULT_IMAGE_TOKEN}\n" + question

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    full_input_text = prompt_question 
    input_ids = tokenizer_image_token(full_input_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    return input_ids


def pad_and_stack(input_ids, pad_token_id, pad_side='left',device=None):

    if input_ids[0].dim() == 2:  # Case 1: (1, seq_len)
        max_len = max(ids.size(1) for ids in input_ids)
        if pad_side == 'left': padded_input_ids = [torch.nn.functional.pad(ids, (max_len - ids.size(1), 0), value=pad_token_id) for ids in input_ids]
        elif pad_side == 'right': padded_input_ids = [torch.nn.functional.pad(ids, (0, max_len - ids.size(1)), value=pad_token_id) for ids in input_ids]
        stacked_input_ids = torch.stack(padded_input_ids)
        return stacked_input_ids.squeeze(1)
    
    elif input_ids[0].dim() == 3:  # Case 2: (1, seq_len, vocab_size)
        max_len = max(ids.size(1) for ids in input_ids)
        # right padding
        pad_tensor = [torch.cat((torch.full((1, max_len - ids.size(1), ids.size(2)), pad_token_id, dtype=ids.dtype).to(ids.device), ids), dim=1) for ids in input_ids]
        stacked_input_ids = torch.stack(pad_tensor)
        return stacked_input_ids.squeeze(1)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)   


def prepare_data(data, tokenizer):
    mixed_data = []
    stay = False
    cnt = 0
    for json, metadata in data:
        cnt += len(json)

        for qa in json:
            question = qa['question']
            answer = qa['answer']
            video_id = qa['video_id']
            d = metadata.get(video_id)
            if d is None: pass
            else:
                input_ids, label_ids, label_ids_length = prepare_input(question, answer, tokenizer)
                if input_ids.shape[1] > 512: continue
                caption = d['caption']
                caption_ids = tokenizer(caption+tokenizer.eos_token, return_tensors='pt')["input_ids"]
                visual_tokens = d['visual_tokens']
                mixed_data.append({
                    'caption': caption_ids,
                    'visual_tokens': visual_tokens,
                    #'question': question,
                    'answer': answer,
                    'input_ids': input_ids,
                    'label_ids': label_ids,
                    'label_ids_length': label_ids_length
                    })

    # Shuffle the mixed data
    #random.shuffle(mixed_data)
    print('Total number of QAs: ', cnt)
    return mixed_data


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        answer = item['answer']
        #question = item['question']
        visual_tokens = item['visual_tokens']
        caption_ids = item['caption']
        input_ids = item['input_ids']
        label_ids = item['label_ids']
        label_ids_length = item['label_ids_length']
        #input_ids, label_ids, label_ids_length = prepare_input(question, answer, self.tokenizer)
        #caption_ids = self.tokenizer(caption+self.tokenizer.eos_token, return_tensors='pt')["input_ids"]
        return {
            'input_ids': input_ids,
            'label_ids': label_ids,
            'label_ids_length': label_ids_length,
            'caption_ids': caption_ids,
            'visual_tokens': visual_tokens,
            'answer': answer
        }
    

class MemoryTreeNode:
    def __init__(self, visual_tokens, caption_ids, children=None):
        
        self.visual_tokens = visual_tokens
        self.caption_ids = caption_ids
        self.children = []
        if children:
            for idx, child in enumerate(children):
                self.children = children

    def add_child(self, child_node):
        self.children.append = child_node

def get_all_memories_in_order(node): # we go from bottom up and from left to right
    children_list = []
    
    # Traverse each child recursively, left to right
    for child in node.children:
        if child:
            children_list.extend(get_all_memories_in_order(child))
    
    # After all children are processed, add the current node
    children_list.append(node)
    
    return children_list

def get_leaf_memories_in_order(node): # we go from bottom up and from left to right

    children_list = []
    # Traverse each child recursively, left to right
    for child in node.children:
        if child:
            children_list.extend(get_all_memories_in_order(child))

    # After all children are processed, add the current node
    if not node.children or all(c is None for c in node.children):
        children_list.append(node)

    return children_list

def base_traverse_tree(node):
    print("Visiting Node with tensor shape:", node.tensor.shape)
    for child in node.children:
        if child is not None:
            base_traverse_tree(child)

# Tree traversal - give root: base_traverse_tree(root)

def frames_from_mp4_path(path, video_name, image_processor):
    if video_name.endswith(('.mp4', '.mkv', '.avi', '.mov')):  # Add video formats as needed
        video_path = os.path.join(path, video_name)
        if not os.path.exists(video_path): 
            print('ERROR: ', video_name)

    print(video_name)
    all_frames, num_total_frames, fps, video_druation = load_video_frames(video_path, 10)
    short_clips = split_into_short_clips_1_fps(all_frames=all_frames, fps=fps, frames_per_short_clip=10)

    image_tensors = []
    image_sizes = []
    for video_frames in short_clips:
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
        image_tensors.append(frames)
        image_size = [frame.size for frame in video_frames]
        image_sizes.append(image_size)

    return image_tensors, image_sizes

'''
def frames_from_images_path(image_dir):
    image_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.png')]
    frames = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        frame = cv2.imread(img_path)
        frame = frame.astype(np.float32)
        frames.append(frame)

    #frames = np.array(frames)
    frames = [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

    image_tensors = []
    image_sizes = []
    for video_frames in short_clips:
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
        image_tensors.append(frames)
        image_size = [frame.size for frame in video_frames]
        image_sizes.append(image_size)

    return image_tensors, image_sizes
'''
def prepare_streaming_data(data, path, source,  image_processor):
    mixed_data = []
    video_dict = {}
    if source == 'moviechat':
        for json in data:
            video_name = json['info']["video_path"]
            image_tensors, image_sizes = frames_from_mp4_path(path, video_name, image_processor)
            questions = []
            answers = []
            for qa in json['global'] + json['breakpoint']:
                questions.append(qa["question"])
                answers.append(qa["answer"])    
            video_dict[video_name] = {"image_tensors": image_tensors, "image_sizes": image_sizes, "questions": questions, "answers":answers}
            '''
            chunks = [(questions[i:i+5], answers[i:i+5]) for i in range(0, len(questions), 5)]
            for idx, (chunk_questions, chunk_answers) in enumerate(chunks):
                key = f"{video_name}_{idx}"
                video_dict[key] = {"image_tensors": image_tensors, "image_sizes": image_sizes, "questions": chunk_questions, "answers":chunk_answers}
            '''

    elif source == 'videoinstruct':
        for d in data:
            video_name = d["video_id"]
            question = d["question"]
            answer = d["answer"]
            if video_dict.get(video_name) is None: 
                image_tensors, image_sizes = frames_from_mp4_path(path, video_name)
                video_dict[video_name] = {"image_tensors": image_tensors, "image_sizes": image_sizes, "questions": [question], "answers":[answer]}
            else:
                video_dict[video_name]["questions"].append(question)
                video_dict[video_name]["answers"].append(answer)

    elif source == 'movieLLM':
        for json, movie in zip(data, path):
            image_tensors, image_sizes = frames_from_images_path(movie)
            questions, answers = [], []
            for qa in json['QA']:
                questions,append(qa['Question'])
                answers.append(qa['Answer'])
            #video_dict[video_name] = {"image_tensors": image_tensors, "image_sizes": image_sizes, "questions": questions, "answers":answers}
            '''
            chunks = [(questions[i:i+5], answers[i:i+5]) for i in range(0, len(questions), 5)]
            for idx, (chunk_questions, chunk_answers) in enumerate(chunks):
                key = f"{video_name}_{idx}"
                video_dict[key] = {"image_tensors": image_tensors, "image_sizes": image_sizes, "questions": chunk_questions, "answers":chunk_answers}
            '''

    elif source == 'tvqa':
        for movie in path:
            image_tensors, image_sizes = frames_from_images_path(movie)
            d = data.get(movie)
            questions = d['questions']
            choices = d['choices']
            answer_indices = d['answer_idx']
            # fix MC format

    else: print('SOURCE ERROR.')

    for video_name, data in video_dict.items():
        mixed_data.append({
            'image_tensors': data["image_tensors"],
            'image_sizes': data["image_sizes"],
            'questions': data["questions"],
            'answers': data["answers"],
            })
    del video_dict
    # Shuffle the mixed data
    #random.shuffle(mixed_data)
    return mixed_data

class StreamingDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_tensors = item['image_tensors']
        image_sizes = item["image_sizes"]
        questions = item["questions"]
        answers = item["answers"]
        input_ids_batch, label_ids_batch, label_ids_length_batch = [], [], []
        for q,a in zip(questions, answers):
            input_ids, label_ids, label_ids_length = prepare_input(q, a, self.tokenizer)
            input_ids_batch.append(input_ids)
            label_ids_batch.append(label_ids)
            label_ids_length_batch.append(label_ids_length)

            input_ids = pad_and_stack(input_ids_batch, self.tokenizer.pad_token_id)
            label_ids = pad_and_stack(label_ids_batch, -100)
            label_ids_length = label_ids_length_batch
   
        return {
            'image_tensors': image_tensors,
            'image_sizes': image_sizes,
            'input_ids': input_ids,
            'label_ids': label_ids,
            'label_ids_length': label_ids_length
        }

import json 
def save_video_data(json_file, video_name, video_data):
    """
    Saves or appends video data to a JSON file, structured by video names.
    Args:
        json_file (str): Path to the JSON file.
        video_name (str): The name of the video (used as a key).
        video_data (dict): The dictionary containing video metadata or information.
    """
    # Check if file exists and read existing data
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}  # If file is empty or corrupted, start fresh
    else:
        data = {}

    # Update the JSON structure with new video data
    data[video_name] = video_data

    # Write the updated data back to the file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def select_top_k_visual_tokens(visual_tokens, top_k_indices):
    #print(top_k_indices)
    #print('###########################')
    #top_k_indices_sorted = torch.sort(top_k_indices).values
    top_k_indices_sorted = sorted(top_k_indices)
    #print(top_k_indices_sorted)
    selected = torch.stack([visual_tokens[:,k-14,:] for k in top_k_indices_sorted], dim=1)
    return selected

def top_k_attention_tokens(attn_scores, start, end, k=5):
    """
    Find the K tokens with the highest average attention score within a given range.
    Parameters:
    - attn_scores: List of 25 lists (generated tokens), each containing 28 layers with a tensor of shape (1,1,fullseqlen).
    - start: Start index of the range in fullseqlen.
    - end: End index of the range in fullseqlen.
    - k: Number of top tokens to return.
    Returns: 
    - List of indices of the top-K tokens with highest average attention.
    """
    num_generated = len(attn_scores)  # 25 generated tokens
    num_layers = len(attn_scores[0])  # 28 layers
    # Stack all attention tensors: Shape (25, 28, 1, 1, fullseqlen)
    attn_tensor = torch.stack([
        torch.stack(layer_attns) for layer_attns in attn_scores
    ])  # Shape: (25, 28, 1, 1, fullseqlen)
    # Remove singleton dimensions and keep only full sequence length
    if attn_tensor.shape[2] == 1: # bs=1
        attn_tensor = attn_tensor.squeeze(dim=(2,3))  # Shape: (25, 28, fullseqlen)
        attn_tensor = attn_tensor[:, :, start:end]
        avg_attn_scores = attn_tensor.mean(dim=(0, 1))
        top_k_indices = torch.topk(avg_attn_scores, k).indices.tolist()
        return [start + idx for idx in top_k_indices]
    else:
        attn_tensor = attn_tensor.squeeze(3).permute(2,0,1,3)  # Shape: (bs, 25, 28, fullseqlen)
        attn_tensor = attn_tensor[:, :, :, start:end]
        #print(attn_tensor.shape)
        # Slice the attention scores within the specified range (start:end)
        # Compute average attention across generated tokens and layers
        avg_attn_scores = attn_tensor.mean(dim=(1, 2))  # Shape: (bs, end-start)
        # Get the top-K indices
        top_k_indices = torch.topk(avg_attn_scores, k).indices #.tolist()
        top_k_indices = [[start + idx for idx in top_k] for top_k in top_k_indices]
        return top_k_indices


def select_visual_tokens(caption, visual_tokens, SELECTION, batch_size, last_clip_num_frames, short_clip_frames=32):
    ### select visual tokens
    if batch_size > 1:
        attentions = [[attn[:-1,-1,:].unsqueeze(1) for attn in caption.attentions[0] if attn is not None]]
        prefill_size = caption.attentions[0][4].shape[1]
        for attn in caption.attentions[1:]:
            attentions.append([a[:-1,:,:prefill_size] for a in attn if a is not None])
        end = 14+(196*short_clip_frames)
        topk_indices = top_k_attention_tokens(attentions,start=14,end=end,k=SELECTION)
        if len(topk_indices) == SELECTION: visual_tokens_list = [select_top_k_visual_tokens(visual_tokens[0].unsqueeze(0), topk_indices)]
        else: visual_tokens_list = [select_top_k_visual_tokens(v.unsqueeze(0), topk) for v, topk in zip(visual_tokens[:-1], topk_indices)]
        #### xxxxxx ####
        ###### end of video ######
    else:
        visual_tokens_list = []
    prefill_size = 14+(196*last_clip_num_frames)+14+1
    attentions = [[attn[-1,-1,:prefill_size].unsqueeze(0).unsqueeze(1) for attn in caption.attentions[0] if attn is not None]]
    #prefill_size = 14+(196*image_tensors[-1].shape[0])+14+1
    for attn in caption.attentions[1:]:
        attentions.append([a[-1,:,:prefill_size].unsqueeze(0) for a in attn if a is not None])
    end = 14+(196*last_clip_num_frames)
    topk_indices = top_k_attention_tokens(attentions,start=14,end=end,k=SELECTION)
    visual_tokens_list.append(select_top_k_visual_tokens(visual_tokens[-1].unsqueeze(0), topk_indices))
    visual_tokens = torch.cat(visual_tokens_list, dim=0) # bs, topk, d
    del visual_tokens_list, attentions
    #### ~~~~~~~~~~~~~~ ####
    return visual_tokens

import cv2
def frames_from_images_path(device,image_dir, image_processor, CHUNK_SIZE, MIN_FRAMES=1):
    image_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg')]
    frames = []
    for img_file in image_files:
        #print(img_file)
        img_path = os.path.join(image_dir, img_file)
        frame = cv2.imread(img_path)
        #plt.imshow(frame)
        #plt.show()
        frame = frame.astype(np.float32)
        frames.append(frame)

    #frames = np.array(frames)
    print('all frames: ', len(frames))
    frames = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(device)
    chunk_size = CHUNK_SIZE
    image_tensors = [frames[i:i+chunk_size] for i in range(0, frames.size(0), chunk_size)]
    ###############
    if image_tensors[-1].size(0) < MIN_FRAMES:
        needed = MIN_FRAMES - image_tensors[-1].size(0)
        pull_from_prev = image_tensors[-2][-needed:]
        image_tensors[-1] = torch.cat((pull_from_prev, image_tensors[-1]), dim=0)
        image_tensors[-2] = image_tensors[-2][:-needed]
    ###############
    image_sizes = [e.size for e in image_tensors]
    return image_tensors, image_sizes

def dededuplicate_sentences(token_ids, dot_id):
    seen = set()
    result = []
    sentence = []
    for token in token_ids:
        sentence.append(token)
        if token == dot_id:
            sentence_tuple = tuple(sentence)
            if sentence_tuple not in seen:
                seen.add(sentence_tuple)
                result.extend(sentence)
            sentence = []

    # Handle last sentence if no trailing dot
    if sentence:
        sentence_tuple = tuple(sentence)
        if sentence_tuple not in seen:
            result.extend(sentence)

    return result

def deduplicate_sentences(captions, dot_id):
    seen = set()
    result = []
    for tokens in captions:  # tokens is a list of token ids (caption)
        new_caption = []
        sentence = []

        for token in tokens:
            sentence.append(token)
            if token == dot_id:
                sentence_tuple = tuple(sentence)
                if sentence_tuple not in seen:
                    seen.add(sentence_tuple)
                    new_caption.extend(sentence)
                sentence = []

        # Handle trailing sentence without a dot
    
        if sentence:
            sentence_tuple = tuple(sentence)
            if sentence_tuple not in seen:
                new_caption.extend(sentence)
        
        if new_caption:
            result.append(new_caption)

    return result

def mmr(query_embedding, doc_embeddings, lambda_param=0.5):
    """
    Computes full Maximal Marginal Relevance (MMR) ranking.
    Args:
        query_embedding (torch.Tensor): shape (d,) or (1, d)
        doc_embeddings (torch.Tensor): shape (n, d)
        lambda_param (float): relevance-diversity trade-off (between 0 and 1)
    Returns:
        List[int]: indices ranked by descending MMR score
    """
    if query_embedding.dim() == 2:
        query_embedding = query_embedding.squeeze(0)  # (d,)

    n = doc_embeddings.size(0)
    selected = []
    remaining = list(range(n))

    # Normalize embeddings
    query_norm = query_embedding / query_embedding.norm()
    doc_norms = doc_embeddings / doc_embeddings.norm(dim=1, keepdim=True)
    sim_to_query = torch.matmul(doc_norms, query_norm)  # (n,)

    while remaining:
        max_score = -float('inf')
        best_idx = -1

        for i in remaining:
            if selected:
                sims_to_selected = torch.matmul(doc_norms[i], doc_norms[selected].T)  # (len(selected),)
                max_sim = sims_to_selected.max().item()
            else:
                max_sim = 0.0

            mmr_score = lambda_param * sim_to_query[i].item() - (1 - lambda_param) * max_sim

            if mmr_score > max_score:
                max_score = mmr_score
                best_idx = i

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected



