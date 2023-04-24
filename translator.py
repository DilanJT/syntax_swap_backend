

# config_path = '../ml_model/graphcodebert/'
# model_path = config_path + 'complete_modelv3.pt'
# tokenizer_path = config_path + 'tokenizerv3.pt'
# model = torch.load(model_path, map_location=torch.device('cpu'))
# tokenizer = torch.load(tokenizer_path, map_location=torch.device('cpu'))
# model.eval()

# def translate(snippet):
#     snippet = snippet.strip()
#     input_ids = tokenizer.encode(snippet, return_tensors='pt')
#     outputs = model(input_ids)[0]
#     translated_ids = torch.argmax(outputs, axis=2).squeeze()
#     translated_text = tokenizer.decode(translated_ids, skip_special_tokens=True)
#     return translated_text

import torch
from tree_sitter import Language, Parser
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import RobertaTokenizer
from tqdm import tqdm, trange
import numpy as np
import torch.nn as nn
from model2 import Seq2Seq
import gc

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

trainned_model4 = "D:\dilan_files\FYP_code\Pytorch_models\syntax_swap_modelv2.pt"
tokenizer = "D:\dilan_files\FYP_code\Pytorch_models\syntax_swap_tokenizerv2.pt"

model = torch.load(trainned_model4, map_location=device)
tokenizer = torch.load(tokenizer, map_location=device)

model.eval()

from DFG import (
    DFG_java,
    DFG_javascript
)
from utils import (
    remove_comments_and_docstrings,
    tree_to_token_index,
    index_to_code_token, 
    tree_to_variable_index
)

dfg_function={
    'java':DFG_java,
    'javascript':DFG_javascript
}

def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    # if lang=="php":
    #     code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg


language_path = './build_parser/my-languages.so'

#load parsers
parsers={}        
for lang in dfg_function:
    print("lang",lang)
    LANGUAGE = Language('./build_parser/java_js.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
print("Parsers :",parsers)

class Example(object):
    """A single training/test example."""
    def __init__(self,source, lang):
        self.source = source
        self.lang = lang

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask  


max_source_length = 320
max_target_length = 256
source_lang = "java"
target_lang = "javascript"

def convert_examples_to_features(examples, tokenizer, stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples,total=len(examples))):
        ##extract data flow
        # check for any error happened in this code
        # check if the data flow extraction is done for both the languages.
        code_tokens,dfg=extract_dataflow(example.source,
                                         parsers["java"],
                                         "java")
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        # ORIGINAL TO CURRENT TOKEN POSITION
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    

        flattened_code_tokens = []
        for x in code_tokens:
            for y in x:
                flattened_code_tokens.append(y)
        code_tokens = flattened_code_tokens
  
        
        #truncating
        code_tokens=code_tokens[:max_source_length-3][:512-3]
        
        # Adds the special tokens [CLS] and [SEP] to the beginning and end of the token 
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        # convert the tokens to their corresponding token ids
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        """
        Creates a position_idx sequence that assigns a unique position index to each token in
        the sequence. The '+1' offset is added to avoid the '0' position index which is reserved 
        for padding.

        position_idx = []
        for i in range(len(source_tokens)):
            position_idx.append(i + tokenizer.pad_token_id + 1)

        """
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg=dfg[:max_source_length-len(source_tokens)]
        """
        Concatenates the dfg edge labels with the source_tokens list and adds a 0 to the
        position_idx list for each dfg edge label.


        """
        source_tokens+=[x[0] for x in dfg]
        """
        Sets the token ids for the dfg edge labels to tokenizer.unk_token_id, indicating 
        that they are unknown tokens.
        """
        position_idx+=[0 for x in dfg]
        source_ids+=[tokenizer.unk_token_id for x in dfg]
        """
        Pads the position_idx and source_ids lists with tokenizer.pad_token_id to ensure 
        that they have the same length as max_source_length.
        """
        padding_length=max_source_length-len(source_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        source_ids+=[tokenizer.pad_token_id]*padding_length  

        """
        Creates a source_mask list that indicates which tokens in the sequence are valid 
        input tokens (1) and which ones are padding tokens (0)
        """
        source_mask = [1] * (len(source_tokens))
        source_mask+=[0]*padding_length       
        
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx

        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    

        
        
        dfg_to_dfg=[x[-1] for x in dfg]

        """
        creates a list dfg_to_code where each element corresponds to a tuple containing 
        the starting and ending positions of the code segment that corresponds to each node 
        in the dataflow graph (dfg). The positions are expressed in terms of the original source code.

        dfg_to_code = []
        for node in dfg:
            # Get the position of the node in the original source code
            position = ori2cur_pos[node[1]]
            # Add the position to the list
            dfg_to_code.append(position)

        
        """
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token]) # length of the special token [CLS], always 1

        """ 
        for x in dfg_to_code:
            new_pos = (x[0] + length, x[1] + length) # Shift the start and end position by length of CLS token
            dfg_to_code.append(new_pos) # Add the new positions to the list
        """
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
      

        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:max_target_length-2]

        
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
   
        if example_index < 5:
            if stage=='train':
                print("*** Example ***")
                print("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                print("source_ids: {}".format(' '.join(map(str, source_ids))))
                print("source_mask: {}".format(' '.join(map(str, source_mask))))
                print("position_idx: {}".format(position_idx))
                print("dfg_to_code: {}".format(' '.join(map(str, dfg_to_code))))
                print("dfg_to_dfg: {}".format(' '.join(map(str, dfg_to_dfg))))
                
                print("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                print("target_ids: {}".format(' '.join(map(str, target_ids))))
                print("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features

class TextDataset(Dataset):
    def __init__(self, examples, max_source_length):
        self.examples = examples
        self.max_source_length = max_source_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.max_source_length,self.max_source_length),dtype=np.bool)
        
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])

        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].source_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes         
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
                    
        return (torch.tensor(self.examples[item].source_ids),
                torch.tensor(self.examples[item].source_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(attn_mask), 
                torch.tensor(self.examples[item].target_ids),
                torch.tensor(self.examples[item].target_mask),)
    

# TODO: change the variable names
def translate(java_code="System.out.println()"):


    if(device == torch.device("cuda")):
        clear_cache()
        example = Example(source=java_code.strip(), lang='java')
        examples = []
        examples.append(example)

        print("Parsers :",parsers)
        code_tokens, dfg=extract_dataflow(example.source, parsers['java'], 'java')
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]

        
        # eval_examples = read_examples(file)
        eval_features = convert_examples_to_features(examples, tokenizer,stage='test')
        eval_data = TextDataset(eval_features, max_source_length) 

        # Calculate bleu
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=8,num_workers=0)

        p=[]
        translated_code = ""
        for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch                    
            with torch.no_grad():
                preds = model(source_ids,source_mask,position_idx,att_mask)
                for pred in preds:
                    t=pred[0].cpu().numpy()
                    t=list(t)
                    print(t)
                    if 0 in t:
                        t=t[:t.index(0)]
                    text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                    translated_code = text
                    p.append(text)
        clear_cache()
        return translated_code

    else:
        return "Can only be translated on a cuda machine. Current device does not support cuda"
    

def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()