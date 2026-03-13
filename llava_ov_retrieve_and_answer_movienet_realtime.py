import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
def setup_args():
    parser = argparse.ArgumentParser('Argparse options for feature extraction pipeline')
    parser.add_argument('--dataset', default='MovieNet-stream', help='Dataset to evaluate.')
    parser.add_argument('--jsons_path', default="...", help='Path to VQA JSONs')
    parser.add_argument('--save_jsons_path', default="new_json_answers/realtime/movienet/", help='Path to GT/generated answers.')
    parser.add_argument('--memories_folder', default="llava_ov_memories/rs_movienet/", help='Path to Video files')
    parser.add_argument('--frames_folder', default="...", help='Path to Video files')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--shuffle', default=False, help='Shuffle the dataloader')
    parser.add_argument('--num_workers', default=4, help='Dataloader workers')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--conv_template', default="qwen_1_5")
    parser.add_argument('--retrieval_mode', default="bp-all-gt", choices=['bp-random-1','bp-top-1','bp-top-10%','bp-random-10%','bp-all-gt','bp-gt'])
    parser.add_argument('--model_size', default="7b", choices=['7b','0.5b'])
    #parser.add_argument('--method', type=str, required=True, help='Method name (required)')
    args, _ = parser.parse_known_args()
    return args
args=setup_args()
from sortedcontainers import SortedDict
from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from tqdm import tqdm
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
import sys 
from utils import get_all_memories_in_order, get_leaf_memories_in_order, deduplicate_sentences, dededuplicate_sentences, mmr
import torch.nn.functional as F
#import seaborn as sns
#import matplotlib.pyplot as plt
import time
warnings.filterwarnings("ignore")
# Load the OneVision model
#pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
##new settings
pretrained= "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
#tokenizer, vlm, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="float32", device_map=device_map, attn_implementation="sdpa", **llava_model_args)
##new setting
tokenizer, vlm, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="float16", device_map=device_map, attn_implementation="flash_attention_2", **llava_model_args)
#vlm = vlm.half()
vlm.eval()
#vlm=vlm.cpu()

from model import rLiVS
import copy

model = rLiVS(vlm, args.device, tokenizer)
    
from utils import pad_and_stack
from eval_loader import get_eval_loader
import random
def get_input_ids(questions):
    # Prepare conversation input
    all_input_ids=[]
    for q in questions:
        question = f"{DEFAULT_IMAGE_TOKEN}\n"+q
        conv = copy.deepcopy(conv_templates[args.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
        all_input_ids.append(input_ids)
    all_input_ids = pad_and_stack(all_input_ids, tokenizer.pad_token_id)
    return all_input_ids

def get_input_ids_sum(captions):
    # Prepare conversation input
    all_input_ids=[]
    for q in questions:
        question = "You are given 4 captions corresponding to 4 subsequent sho"+q
        conv = copy.deepcopy(conv_templates[args.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
        all_input_ids.append(input_ids)
    all_input_ids = pad_and_stack(all_input_ids, tokenizer.pad_token_id)
    return all_input_ids

def get_input_ids_2(questions):
    # Prepare conversation input
    all_input_ids=[]
    for q in questions:
        question = copy.deepcopy(q)
        input_ids = tokenizer(question+tokenizer.eos_token, return_tensors='pt')["input_ids"]
        all_input_ids.append(input_ids)
    all_input_ids = pad_and_stack(all_input_ids, tokenizer.pad_token_id)
    return all_input_ids

#Retireve Memories for General/BP Question-answering
def memory_retrieval(args, cut_memories, caption_ids, input_ids, model,vid_start_shot=None,vid_end_shot=None,gt_start_shot=None,gt_end_shot=None, topk=None):
    num_queries=input_ids.shape[0]
    avg_caption_ids=caption_ids.mean(dim=2).to(args.device)

    if args.retrieval_mode=="bp-all-gt":
        MAX_LENGTH = 20000
        num_memories = cut_memories.shape[1]
        one_mem_size = cut_memories.shape[2]
        if (num_memories*one_mem_size) < MAX_LENGTH:
            return cut_memories, None
        else:
            max_fit_memories = int(MAX_LENGTH  // one_mem_size)
            argss = copy.deepcopy(args)
            argss.retrieval_mode='bp-top-10%'
            return memory_retrieval(argss, cut_memories, caption_ids, input_ids, model, topk=max_fit_memories)
    elif args.retrieval_mode=='bp-gt':
        #import pdb;pdb.set_trace()
        if args.dataset=="MovieNet-stream":
            gt_start_offset_frames=(gt_start_shot - vid_start_shot) * 3
            gt_end_offset_frames= (vid_end_shot - gt_end_shot) *3
        else:
            gt_start_offset_frames=(gt_start_shot - vid_start_shot)
            gt_end_offset_frames= (vid_end_shot - gt_end_shot)      
        gt_mem_start_idx =  (gt_start_offset_frames//32)
        gt_mem_end_idx= cut_memories.shape[1] - (gt_end_offset_frames//32)
        cut_memories= copy.deepcopy(cut_memories[:,gt_mem_start_idx:gt_mem_end_idx])
        caption_ids= copy.deepcopy(caption_ids[:,gt_mem_start_idx:gt_mem_end_idx])
        #######
        argss = copy.deepcopy(args)
        argss.retrieval_mode='bp-all-gt'
        return memory_retrieval(argss, cut_memories, caption_ids, input_ids, model)
        #return cut_memories, np.arange(gt_mem_start_idx,gt_mem_end_idx)
    elif args.retrieval_mode=='bp-top-10%':
        if topk==None: top_k=max(1,cut_memories.shape[1]//10)
        else: top_k = topk 
        device=model.vlm.model.embed_tokens.weight.device
        query_embedding=model.vlm.model.embed_tokens(input_ids.to(device))
        query_embedding_avg=query_embedding.mean(dim=1)
        
        #import pdb;pdb.set_trace()
        # Normalize A and B for cosine similarity
        query_embedding_avg_norm = query_embedding_avg / query_embedding_avg.norm(dim=1, keepdim=True)  # (5, 896)
        avg_cut_memories_norm = avg_caption_ids / avg_caption_ids.norm(dim=2, keepdim=True) # (5, 45, 896)

        cos_sim = torch.einsum('ij,ikj->ik', query_embedding_avg_norm.float(), avg_cut_memories_norm.float())

        # Get top-k indices and values
        top_values, top_indices = cos_sim.topk(top_k, dim=1)  # (5, 3)

        # Retrieve the top-k best-matching vectors (5, k, 896)
        sorted_indices = torch.sort(top_indices, dim=1).values
        best_memories = cut_memories[torch.arange(num_queries).unsqueeze(1), sorted_indices]
        return best_memories, np.array(top_indices.tolist()).reshape(-1)
    elif args.retrieval_mode=="bp-random-1":
        up_idx=cut_memories.shape[1]
        random_indice = random.choices(range(0, up_idx), k=num_queries)
        best_memories = cut_memories[torch.arange(num_queries), random_indice].unsqueeze(1)
        return best_memories, random_indice
    elif args.retrieval_mode=="bp-top-1":
        device=model.vlm.model.embed_tokens.weight.device
        query_embedding=model.vlm.model.embed_tokens(input_ids.to(device))
        query_embedding_avg=query_embedding.mean(dim=1)
        # Normalize A and B for cosine similarity
        query_embedding_avg_norm = query_embedding_avg / query_embedding_avg.norm(dim=1, keepdim=True)  # (5, 896)
        avg_cut_memories_norm = avg_cut_memories / avg_cut_memories.norm(dim=2, keepdim=True) # (5, 45, 896)
        cos_sim = torch.einsum('ij,ikj->ik', query_embedding_avg_norm.float(), avg_cut_memories_norm.float())

        # Get the index of max similarity (5,)
        max_indices = cos_sim.argmax(dim=1)
        # Retrieve the best-matching vectors (5, 896)
        best_memories = cut_memories[torch.arange(num_queries), max_indices].unsqueeze(1)
        return best_memories, np.array(max_indices.tolist()).reshape(-1)
    elif args.retrieval_mode=='bp-random-10%':
        top_k=max(1,cut_memories.shape[1]//10)
        up_idx=cut_memories.shape[1]
        random_indice=np.array([random.choices(range(0, up_idx), k=top_k) for _ in range(num_queries)])
        best_memories = cut_memories[torch.arange(num_queries).unsqueeze(1), random_indice]
        return best_memories, random_indice.reshape(-1)
        
#Break-point Questions answering
import einops
from utils import save_video_data
args.memories_folder = os.path.join(args.memories_folder)
### check if args.method is correct
if not os.path.exists(args.memories_folder):
    print(f"ERROR: Folder does not exist inside {args.memories_folder}!")
    sys.exit(1)
###
f = os.path.join(args.save_jsons_path)
save_folder=os.path.join(f,args.retrieval_mode+'/')
os.makedirs(save_folder, exist_ok=True)
json_save_path=os.path.join(save_folder,args.dataset+'-rlivs_mmr-20k.json')
data_loader=get_eval_loader(args)
invalid_answers = 0
correct_answers = 0 
total_answers = 0
with torch.no_grad():
    #for id_batch, data_batch in enumerate(data_loader):
    for id_batch, data_batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Loading batches"):
        #print('Video #{}'.format(id_batch))
        video_names, memory_tokens, questions, answers, start_times, end_times, vids_start_time,vids_end_time = data_batch  
        for b in range(args.batch_size):
            new_video_dict={}
            video_name=video_names[b]
            preds=[]
            pred_mem_idxs=[]
            if True: #for q in range(len(questions[b])):
                qs = 1 #this_input_ids.shape[0]
                all_memories = []

                for node in memory_tokens[b]:
                    all_memories += get_all_memories_in_order(node)
                    #all_memories += get_leaf_memories_in_order(node)
                
                visual_list, textual_list = [], []
                global_idx_counter = 0
                visual_idx_list = []
                cap_list = []
                for e in all_memories:
                    v = e.visual_tokens
                    t = e.caption_ids
                    if isinstance(v, np.ndarray): v = torch.from_numpy(v)
                    v = v.to(args.device)
                    visual_list.append(v)
                    if isinstance(t, np.ndarray): t = torch.from_numpy(t)
                    cap_list.append(t)
                
                dot_id = tokenizer('.', return_tensors='pt')["input_ids"]
                flattened_ids = [t[0].tolist() for t in cap_list]
                #textual_list = deduplicate_sentences(flattened_ids, dot_id[0])
                textual_list = flattened_ids 
                ##########
                '''
                deduplcated_caps = [tokenizer.batch_decode(torch.tensor(sublist).unsqueeze(0), skip_special_tokens=True) for sublist in textual_list]
                
                _, ans = model.generate_vlm(input_ids=this_input_ids, images=this_memory_tokens.half(), image_sizes=None,
                                                          input_is_memory=True, max_new_tokens=1024, get_logits=False,
                                                             modalities=["video" for _ in range(qs)])
                '''
                ##########
                cap_sizes = [len(t) for t in textual_list]
                total_length = sum(cap_sizes)
                device=model.vlm.model.embed_tokens.weight.device
                textual_list = [model.vlm.model.embed_tokens(torch.tensor(sublist).unsqueeze(0).to(device)) for sublist in textual_list] 
                all_caption_embs = pad_and_stack(textual_list, 0).expand(qs, -1, -1, -1) # we pad with 0-tensors so it doesnt affect the cosine sim
                all_memory_tokens = torch.cat([cap for cap in textual_list], dim=1)
                mmr_embs = torch.stack([e.mean(dim=1).squeeze(0) for e in textual_list], dim=0)
                #print('mmr embs shape: ', mmr_embs.shape)
            for q in range(len(questions[b])):
                this_input_ids = get_input_ids([questions[b][q]])
                #############################################################################
                #Get questions embeddings with th q-former dimension for similarity matching
                question_ids=get_input_ids_2([questions[b][q]])
                if total_length <= 20000:
                    #print('fits')
                    this_memory_tokens = all_memory_tokens #torch.cat([cap for cap in textual_list], dim=1)
                else:
                    '''
                    model.vlm = model.vlm.to('cpu')
                    ######
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    start_memory = torch.cuda.memory_allocated()
                    start_peak = torch.cuda.max_memory_allocated()
                    model.vlm = model.vlm.cuda()
                    #####
                    '''
                    start = time.time() 
                    ###### 1. Similarity #####
                    avg_caption_embs=all_caption_embs.mean(dim=2).to(args.device)
                    device=model.vlm.model.embed_tokens.weight.device
                    query_embedding=model.vlm.model.embed_tokens(question_ids.to(device))
                    query_embedding_avg=query_embedding.mean(dim=1)

                    #start = time.time()
                    all_indices_ordered = mmr(query_embedding_avg, mmr_embs)
                    ordered_dict = SortedDict()
                    current_context = 0
                    for next_most_similar_index in all_indices_ordered:
                        current_context += cap_sizes[next_most_similar_index]
                        if current_context >= 20000:
                            break
                        else:
                            ordered_dict[next_most_similar_index] = textual_list[next_most_similar_index]
                    
                    this_memory_tokens = torch.cat([cap for cap in ordered_dict.values()], dim=1)

                _, ans = model.generate_vlm(input_ids=this_input_ids, images=this_memory_tokens.half(), image_sizes=None, 
                                                          input_is_memory=True, max_new_tokens=256, get_logits=False,
                                                            modalities=["video" for _ in range(qs)])
                
                end = time.time()
                inference_time = end - start
                #end_memory = torch.cuda.memory_allocated()
                #peak_memory = torch.cuda.max_memory_allocated()
                #print(f"Used memory: {(end_memory - start_memory) / (1024 ** 2):.2f} MB")
                #print(f"Peak memory during block: {(peak_memory - start_memory) / (1024 ** 2):.2f} MB")
                print(f"Inference time: {inference_time:.3f} seconds")
                
                pred = tokenizer.batch_decode(ans.sequences, skip_special_tokens=True)
                # print('Question: ', questions[b][q])
                # print('Answer: ', pred)
                preds.extend(pred)
                #pred_mem_idxs.append(pred_memory_idx)
                torch.cuda.empty_cache()
            del textual_list, all_caption_embs, all_memory_tokens
            #sys.exit()
            for p, pred_answer in enumerate(preds):
                new_video_dict[str(p)]={}
                new_video_dict[str(p)]['question'] = copy.deepcopy(questions[b][p])
                new_video_dict[str(p)]['gt'] = copy.deepcopy(answers[b][p])
                new_video_dict[str(p)]['mem_pred'] = copy.deepcopy(pred_answer)
                #new_video_dict[str(p)]['gt_memory_idx'] = timestamps[b][p]//300
                #if len(pred_mem_idxs[p])==1:
                #    new_video_dict[str(p)]['pred_memory_idx'] = str(pred_mem_idxs[p][0])
                #else:
                #    try:
                #        new_video_dict[str(p)]['pred_memory_idx']= ','.join(map(str, pred_mem_idxs[p].astype(int)))
                #    except:
                #        if pred_memory_idx==None:
                #            new_video_dict[str(p)]['pred_memory_idx'] = 'ALL'

            save_video_data(json_save_path, video_name, new_video_dict)
            
    
