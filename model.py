import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import torch
import einops
from utils import pad_and_stack, MemoryTreeNode, get_all_memories_in_order
import copy
from collections import deque
from itertools import chain
import torch.nn.functional as F

class rLiVS(nn.Module):
    def __init__(self, vlm, device, tokenizer):
        super(rLiVS, self).__init__()
        
        self.tokenizer = tokenizer
        self.device = device
        # LLaVA-OV
        self.vlm = vlm
 
    def forward_vlm(self, input_ids, memory_tokens, labels, image_sizes=None, modalities=["video"]):
        
            output = self.vlm.custom_forward(
                input_is_memory=True,
                input_ids=input_ids,
                images=memory_tokens,
                image_sizes=image_sizes,
                labels=labels,
                modalities=modalities,
            )  

            #print('Caption shape: ', caption.sequences.shape)
            return output
     
    def recurrent_generate_vlm(self, input_ids, image_tensors, image_sizes, previous_memories, output_hidden_states=False, max_new_tokens=1024, get_logits=False, modalities=["video"], attentions=False):

        input_ids = input_ids.cuda()
        if torch.is_tensor(image_tensors): image_tensors = image_tensors.cuda()
        if previous_memories is not None: previous_memories=previous_memories.cuda()
        visual_tokens, caption = self.vlm.recurrent_custom_generate(
            memory_tokens=previous_memories,
            inputs=input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
            modalities=modalities,
            get_logits=get_logits,
            output_hidden_states=output_hidden_states,
            attentions=attentions
        )  
  
        return visual_tokens, caption
    ######################################################################################
  
    def generate_vlm(self, input_ids, images, image_sizes, input_is_memory, output_hidden_states=False, max_new_tokens=1024, get_logits=False, modalities=["video"], attentions=False):
      
        # Step 1: Full pass through the VLM (get caption and visual tokens)
        visual_tokens, caption = self.vlm.custom_generate(
            input_is_memory=input_is_memory,
            inputs=input_ids,
            images=images,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
            modalities=modalities,
            get_logits=get_logits,
            output_hidden_states=output_hidden_states,
            attentions=attentions
        )  
        '''
        visual tokens: (bs, seq_len, D)
        caption: (bs, seq_len) -> ids
        '''
        
        return visual_tokens, caption 
   
    def random_caption_and_select(self, image_tensors, image_sizes, past_context, input_is_memory, start_of_selection=0, SELECTION=196):
        # Prepare conversation input
        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        bs = 1
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        modalities = ["video" for _ in range(bs)]
        if input_is_memory:
            visual_tokens, caption = self.generate_vlm(input_is_memory=True,input_ids=input_ids,
                                                       images=image_tensors, image_sizes=None, modalities=modalities, attentions=False)

        else:
            #if past_context is not None: print('past context: ', past_context.shape)
            visual_tokens, caption = self.recurrent_generate_vlm(input_ids, image_tensors=[image_tensors], image_sizes=[image_sizes], previous_memories=past_context, modalities=modalities, attentions=False)
        #print('vistok shape: ', visual_tokens.shape)
        past_context_size = 0 if past_context is None else past_context.shape[1]
        if past_context_size == 0:
            indices = torch.randint(0, visual_tokens.size(1), (visual_tokens.size(0), SELECTION))
        else:
            start=past_context_size
            indices = torch.randint(start, visual_tokens.size(1), (visual_tokens.size(0), SELECTION))
        indices, _ = torch.sort(indices, dim=1)
        visual_tokens = torch.gather(visual_tokens, 1, indices.unsqueeze(-1).expand(-1, -1, visual_tokens.size(-1)).to(visual_tokens.device))
        #print('selected - vistok shape: ', visual_tokens.shape)
        del indices
        return visual_tokens, caption.sequences

    def caption_and_select(self, image_tensors, image_sizes, past_context, input_is_memory, start_of_selection=0, SELECTION=196): 
        # Prepare conversation input
        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        bs = 1
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        modalities = ["video" for _ in range(bs)]
        if input_is_memory:
            visual_tokens, caption = self.generate_vlm(input_is_memory=True,input_ids=input_ids,
                                                        images=image_tensors, image_sizes=None, modalities=modalities, attentions=True)
        else:
            #if past_context is not None: print('past context: ', past_context.shape)
            visual_tokens, caption = self.recurrent_generate_vlm(input_ids, image_tensors=[image_tensors], image_sizes=[image_sizes], previous_memories=past_context, modalities=modalities, attentions=True)
        #print('visual tokens: ', visual_tokens.shape)
        ### select visual tokens
        '''
        attentions = [[attn[:-1,-1,:].unsqueeze(1) for attn in caption.attentions[0] if attn is not None]]
        prefill_size = caption.attentions[0][4].shape[1]
        for attn in caption.attentions[1:]:
            attentions.append([a[:-1,:,:prefill_size] for a in attn if a is not None])
        end = 14+(196*image_tensors[0].shape[0])
        topk_indices = top_k_attention_tokens(attentions,start=14+start_of_selection,end=end,k=196)
        if len(topk_indices) == 196: visual_tokens_list = [select_top_k_visual_tokens(visual_tokens[0].unsqueeze(0), topk_indices)]
        else: visual_tokens_list = [select_top_k_visual_tokens(v.unsqueeze(0), topk) for v, topk in zip(visual_tokens[:-1], topk_indices)]
        '''
        #### xxxxxx ####
        past_context_size = 0 if past_context is None else past_context.shape[1]
        prefill_size = 14+(196*image_tensors.shape[0])+past_context_size+14+1
        attentions = [[attn[-1,-1,:prefill_size].unsqueeze(0).unsqueeze(1) for attn in caption.attentions[0] if attn is not None]]
        #prefill_size = 14+(196*image_tensors[-1].shape[0])+14+1
        for attn in caption.attentions[1:]:
            attentions.append([a[-1,:,:prefill_size].unsqueeze(0) for a in attn if a is not None])
        end = 14+(196*image_tensors.shape[0])+past_context_size
        topk_indices = top_k_attention_tokens(attentions,start=14+past_context_size,end=end,k=SELECTION)
        '''
        if image_tensors.shape[0] == 1: topk_indices = top_k_attention_tokens(attentions,start=14+past_context_size,end=end,k=196)
        else: topk_indices = top_k_attention_tokens(attentions,start=14+past_context_size,end=end,k=SELECTION)
        '''
        visual_tokens_list = [select_top_k_visual_tokens(visual_tokens[-1].unsqueeze(0), topk_indices)]

        visual_tokens = torch.cat(visual_tokens_list, dim=0)
        return visual_tokens, caption.sequences

    def recurrent_random_simple(self, images, image_sizes, short_term_mem_size, SELECTION):
        memory_pool = []
        past_context = deque(maxlen=short_term_mem_size)
        #############
        for img_tnsr, img_siz in zip(images, image_sizes):
            #print('') 
            # STEP 1: process new frames
            if len(past_context) == 0: full_past_context = None
            else:  
                full_past_context = torch.cat([n.visual_tokens for n in past_context], dim=1)
            
            with torch.no_grad():
                selected_tokens, caption_ids = self.random_caption_and_select(image_tensors=img_tnsr, image_sizes=img_siz,  past_context=full_past_context, input_is_memory=False, SELECTION=SELECTION)

            if len(past_context) == past_context.maxlen :
                for_memory_pool = past_context.popleft()
                memory_pool.append(for_memory_pool)
            node = MemoryTreeNode(selected_tokens, caption_ids)
            past_context.append(node)

        while len(past_context) > 0:
            for_memory_pool = past_context.popleft()
            memory_pool.append(for_memory_pool)
        return memory_pool

    def recurrent_simple(self, images, image_sizes, short_term_mem_size, SELECTION):
        memory_pool = []
        past_context = deque(maxlen=short_term_mem_size)
        #############
        for img_tnsr, img_siz in zip(images, image_sizes):
            #print('') 
            # STEP 1: process new frames
            if len(past_context) == 0: full_past_context = None
            else:
                full_past_context = torch.cat([n.visual_tokens for n in past_context], dim=1)
            with torch.no_grad():
                selected_tokens, caption_ids = self.caption_and_select(image_tensors=img_tnsr, image_sizes=img_siz,  past_context=full_past_context, input_is_memory=False, SELECTION=SELECTION)
            print('selected tokens shape: ', selected_tokens.shape)
            if len(past_context) == past_context.maxlen :
                for_memory_pool = past_context.popleft()
                memory_pool.append(for_memory_pool)
            node = MemoryTreeNode(selected_tokens, caption_ids)
            past_context.append(node)

        while len(past_context) > 0:
            for_memory_pool = past_context.popleft()
            memory_pool.append(for_memory_pool)
        return memory_pool

##############################################################################
##############################################################################

def get_most_similar(current_visual_tokens, memory_pool, k):

    query = current_visual_tokens.mean(dim=1)
    flat_memory = torch.cat([m.visual_tokens for m in memory_pool], dim=0) #(num_memories, memlen,d)    
    avg_flat_memory = flat_memory.mean(dim=1) #(num_memories,,d
    query_norm = F.normalize(query, dim=-1)         # shape: (1, d)
    memory_norm = F.normalize(avg_flat_memory, dim=-1)  # shape: (num_memories, d)
    similarities = torch.matmul(memory_norm, query_norm.T).squeeze(1)  # shape: (num_memories,)
    k_ = min(k, similarities.size(0))
    topk_vals, topk_indices = torch.topk(similarities, k=k_, dim=0)

    sorted_top_indices = torch.sort(topk_indices).values
    #print('shape:  ', flat_memory[sorted_top_indices].unsqueeze(0).shape)
    return flat_memory[sorted_top_indices].reshape(1, -1, flat_memory.size(-1)) 

def select_top_k_visual_tokens(visual_tokens, top_k_indices):
    top_k_indices_sorted = sorted(top_k_indices)
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
    #print(attn_tensor.shape)
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


