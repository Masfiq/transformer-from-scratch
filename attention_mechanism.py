import sys
print("python:", sys.version.split()[0])
import torch
import torch.nn as nn
import torch.optim as optim
import copy

import torchtext # type: ignore
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator # type: ignore
print(torch.__version__)
print(torchtext.__version__)
print(torch.backends.mps.is_available()) 

from math import ceil

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

from collections import Counter

from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq
from feature_extractor import FeatureExtractor

class AttentionMechanism():
    def __init__(self):
        # read the data
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
        self.val_sentence_ids = {}
        self.test_sentence_ids = {}
        
        self.device = 'cpu'
                
        self.vocab = None
        self.text_pipeline = None
        
        self.best_valid_loss = float('inf')
        
    def build_model(self):
        self.INPUT_DIM = len(self.vocab)
        self.OUTPUT_DIM = len(self.vocab)
        self.HID_DIM = 256
        self.ENC_LAYERS = 3
        self.DEC_LAYERS = 3
        self.ENC_HEADS = 8
        self.DEC_HEADS = 8
        self.ENC_PF_DIM = 512
        self.DEC_PF_DIM = 512
        self.ENC_DROPOUT = 0.1
        self.DEC_DROPOUT = 0.1
        
        self.enc = Encoder(self.INPUT_DIM,
                    self.HID_DIM,
                    self.ENC_LAYERS,
                    self.ENC_HEADS,
                    self.ENC_PF_DIM,
                    self.ENC_DROPOUT,
                    self.device)

        self.dec = Decoder(self.OUTPUT_DIM,
                    self.HID_DIM,
                    self.DEC_LAYERS,
                    self.DEC_HEADS,
                    self.DEC_PF_DIM,
                    self.DEC_DROPOUT,
                    self.device)
        
        self.PAD_IDX = self.vocab.get_stoi()['<pad>']
        self.MASK_IDX = self.vocab.get_stoi()['<mask>']
        self.MASK_PERCENT = .15
        
        self.s2s = Seq2Seq(self.enc, self.dec, self.PAD_IDX, self.PAD_IDX, self.device).to(self.device)
        
        self.LEARNING_RATE = 0.0005
        
        self.optimizer = torch.optim.Adam(self.s2s.parameters(), lr = self.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        
    def process_batch(self,batch, mask_idx, mask_perc, max_len=200):
        text_list = []
        masked_text_list = []
        for _text in batch:
            text_holder = torch.ones(max_len, dtype=torch.int32) # fixed size tensor of max_len
            masked_text_holder = torch.ones(max_len, dtype=torch.int32) # fixed size tensor of max_len
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int32)
            
            # idxs_to_mask = sorted(np.random.choice(len(processed_text), ceil(mask_perc*len(processed_text)), replace=False))
            #my code begins
            idxs_to_mask = torch.tensor(
                np.random.choice(len(processed_text), ceil(mask_perc * len(processed_text)), replace=False),
                dtype=torch.int64
            )
            #my code ends
            masked_text = copy.deepcopy(processed_text)
            masked_text[idxs_to_mask] = mask_idx
            pos = min(200, len(processed_text))
            text_holder[-pos:] = processed_text[-pos:]
            masked_text_holder[-pos:] = masked_text[-pos:]
            text_list.append(text_holder.unsqueeze(dim=0))
            masked_text_list.append(masked_text_holder.unsqueeze(dim=0))
        return torch.cat(masked_text_list, dim=0), torch.cat(text_list, dim=0)

    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    def initialize_weights(self,m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
            
    def train(self, model, masked_text, unmasked_text, clip):
        self.s2s.train()
                
        src = masked_text
        trg = unmasked_text
        
        self.optimizer.zero_grad()
        
        output, _ = self.s2s(src, trg)
                        
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]

        output = torch.softmax(output, axis=2)
        trg = nn.functional.one_hot(trg.long(),num_classes=output_dim).to(torch.float)
        
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
        trg = trg.to(self.device)
                    
        loss = self.criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.s2s.parameters(), clip)
        
        self.optimizer.step()
                    
        return loss.item()

    def evaluate(self, model, masked_text, unmasked_text):
        self.s2s.eval()
                
        with torch.no_grad():
            src = masked_text
            trg = unmasked_text

            output, _ = self.s2s(src, trg)
            layer_name = "encoder.layers.2.positionwise_feedforward.fc_2"
            emb = FeatureExtractor(self.s2s, layers=[layer_name])
            embeddings = emb(src, trg)[layer_name]
            
            self.check_embeddings_val(embeddings)

            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output_probs = torch.softmax(output, axis=2)
            trg = nn.functional.one_hot(trg.long(),num_classes=output_dim).to(torch.float).to(self.device)
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = self.criterion(output_probs, trg)
            
        return loss.item()
        
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
        
    def check_embeddings_val(self,embeddings):
        
        
         # ---- STEP 1: Choose 10 focus words automatically from validation data ----
        # We scan sentences to find tokens that appear at least twice somewhere (so we can compare)
        token_counts = Counter()
        for sid in self.val_sentence_ids:
            token_counts.update(self.val_sentence_ids[sid])

        # pick 10 words that appear at least twice in the validation dataset
        focus_words = [w for w, c in token_counts.items() if c >= 2 and w.isalpha()]
        focus_words = focus_words[:10]

        print("\n---- VALIDATION EMBEDDING ANALYSIS ----")
        print(f"Focus words: {focus_words}\n")

        # ---- STEP 2: For each focus word, use where_is to find occurrences ----
        for word in focus_words:
            locations = self.where_is(word, self.val_sentence_ids)

            # flatten (sentence_id, index) pairs
            occurrences = [(sid, pos)
                        for sid, pos_list in locations.items()
                        for pos in pos_list]

            # need at least 2 occurrences to compare
            if len(occurrences) < 2:
                continue

            print(f"\n=== WORD: {word} ===")

            # ---- CASE 1: Same word within a single sentence ----
            # find any sentence with at least 2 occurrences
            same_sentence_pairs = []
            for sid, pos_list in locations.items():
                if len(pos_list) >= 2:
                    same_sentence_pairs.append((sid, pos_list[0], pos_list[1]))
                    break
            
            if same_sentence_pairs:
                sid, p1, p2 = same_sentence_pairs[0]
                e1 = embeddings[sid][p1]
                e2 = embeddings[sid][p2]
                print(f"Same sentence: '{word}' at ({sid},{p1}) vs ({sid},{p2}) ->",
                    self.cosine_sim(e1, e2))

            # ---- CASE 2: Same word across different sentences ----
            sid1, pos1 = occurrences[0]
            sid2, pos2 = occurrences[1]
            e1 = embeddings[sid1][pos1]
            e2 = embeddings[sid2][pos2]
            print(f"Different sentences: '{word}' at ({sid1},{pos1}) vs ({sid2},{pos2}) ->",
                self.cosine_sim(e1, e2))

            # ---- CASE 3: Focus word vs a random other word in same sentence ----
            # pick a random word position from the same sentence as sid1
            sentence_tokens = self.val_sentence_ids[sid1]
            if len(sentence_tokens) > 3:
                other_pos = random.randint(0, len(sentence_tokens) - 1)
                e_other = embeddings[sid1][other_pos]
                other_word = sentence_tokens[other_pos]
                print(f"Word '{word}' ({sid1},{pos1}) vs random word '{other_word}' ({sid1},{other_pos}) ->",
                    self.cosine_sim(e1, e_other))

        
        # print(self.val_sentence_ids[1][8],self.val_sentence_ids[1][11],self.cosine_sim(embeddings[1][8],embeddings[1][11]))
        # print(self.val_sentence_ids[1][8],self.val_sentence_ids[2][42],self.cosine_sim(embeddings[1][8],embeddings[2][42]))
        # print(self.val_sentence_ids[1][8],self.val_sentence_ids[1][100],self.cosine_sim(embeddings[1][8],embeddings[1][100]))
        # print(self.val_sentence_ids[1][100],self.val_sentence_ids[2][42],self.cosine_sim(embeddings[1][100],embeddings[2][42]))
        
        # print(self.val_sentence_ids[24][1],self.val_sentence_ids[24][29],self.cosine_sim(embeddings[24][1],embeddings[24][29]))
        # print(self.val_sentence_ids[24][1],self.val_sentence_ids[24][14],self.cosine_sim(embeddings[24][1],embeddings[24][14]))
        # print(self.val_sentence_ids[24][1],self.val_sentence_ids[26][60],self.cosine_sim(embeddings[24][1],embeddings[26][60]))
        # print(self.val_sentence_ids[24][14],self.val_sentence_ids[26][60],self.cosine_sim(embeddings[24][14],embeddings[26][60]))
        
        # print(self.val_sentence_ids[55][6],self.val_sentence_ids[55][29],self.cosine_sim(embeddings[55][6],embeddings[55][29]))
        # print(self.val_sentence_ids[55][6],self.val_sentence_ids[55][46],self.cosine_sim(embeddings[55][6],embeddings[55][46]))
        # print(self.val_sentence_ids[55][6],self.val_sentence_ids[56][58],self.cosine_sim(embeddings[55][6],embeddings[56][58]))
        # print(self.val_sentence_ids[55][6],self.val_sentence_ids[57][110],self.cosine_sim(embeddings[55][6],embeddings[57][110]))
        
        # print(self.val_sentence_ids[55][1],self.val_sentence_ids[55][6],self.cosine_sim(embeddings[55][1],embeddings[55][6]))
        # print(self.val_sentence_ids[55][1],self.val_sentence_ids[55][32],self.cosine_sim(embeddings[55][1],embeddings[55][32]))
        # print(self.val_sentence_ids[55][1],self.val_sentence_ids[56][20],self.cosine_sim(embeddings[55][1],embeddings[56][20]))
        # print(self.val_sentence_ids[55][1],self.val_sentence_ids[57][110],self.cosine_sim(embeddings[55][1],embeddings[57][110]))
    
    # def check_embeddings_test(self):
    #     pass
    
    def check_embeddings_test(self):
        print("\n---- TEST EMBEDDING ANALYSIS ----")

        # ----- STEP 1: Build test embeddings (same method as in evaluate) -----
        test_src, test_trg = self.process_batch(self.test_data,
                                                self.MASK_IDX,
                                                self.MASK_PERCENT)

        self.s2s.eval()
        with torch.no_grad():
            output, _ = self.s2s(test_src, test_trg)
            layer_name = "encoder.layers.2.positionwise_feedforward.fc_2"
            emb_extractor = FeatureExtractor(self.s2s, layers=[layer_name])
            embeddings = emb_extractor(test_src, test_trg)[layer_name]

        # ----- STEP 2: Choose 10 focus words from test_data -----
        token_counts = Counter()
        for sid in self.test_sentence_ids:
            token_counts.update(self.test_sentence_ids[sid])

        # pick words that appear at least twice
        focus_words = [w for w, c in token_counts.items() if c >= 2 and w.isalpha()]
        focus_words = focus_words[:10]

        print(f"Focus words (test): {focus_words}\n")

        # ----- STEP 3: Analyze each focus word -----
        for word in focus_words:

            locations = self.where_is(word, self.test_sentence_ids)

            occurrences = [(sid, pos)
                        for sid, pos_list in locations.items()
                        for pos in pos_list]

            if len(occurrences) < 2:
                continue

            print(f"\n=== WORD: {word} ===")

            # CASE 1: Same word within a single sentence
            same_sentence_pairs = []
            for sid, pos_list in locations.items():
                if len(pos_list) >= 2:
                    same_sentence_pairs.append((sid, pos_list[0], pos_list[1]))
                    break

            if same_sentence_pairs:
                sid, p1, p2 = same_sentence_pairs[0]
                e1 = embeddings[sid][p1]
                e2 = embeddings[sid][p2]
                print(f"Same sentence: '{word}' at ({sid},{p1}) vs ({sid},{p2}) ->",
                    self.cosine_sim(e1, e2))

            # CASE 2: Same word in different sentences
            (sid1, pos1), (sid2, pos2) = occurrences[0], occurrences[1]
            e1 = embeddings[sid1][pos1]
            e2 = embeddings[sid2][pos2]
            print(f"Different sentences: '{word}' at ({sid1},{pos1}) vs ({sid2},{pos2}) ->",
                self.cosine_sim(e1, e2))

            # CASE 3: Compare to random other word in same sentence
            sentence_tokens = self.test_sentence_ids[sid1]
            if len(sentence_tokens) > 3:
                other_pos = random.randint(0, len(sentence_tokens) - 1)
                other_word = sentence_tokens[other_pos]
                e_other = embeddings[sid1][other_pos]
                print(f"Word '{word}' ({sid1},{pos1}) vs random '{other_word}' ({sid1},{other_pos}) ->",
                    self.cosine_sim(e1, e_other))

        
    # define a cosine similarity function
    # def cosine_sim(self, a, b):
    #     a = a.detach().cpu()
    #     b = b.detach().cpu()
    #     return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    #my code begins
    def cosine_sim(self, a, b):
        # a and b are torch tensors: [hidden_dim]
        a = a.detach().cpu().float()
        b = b.detach().cpu().float()
        
        dot = torch.dot(a, b)
        norm_a = torch.norm(a, p=2)
        norm_b = torch.norm(b, p=2)
        
        return (dot / (norm_a * norm_b + 1e-8)).item()
    #my code ends
        
    # define a function that will tell me where a token occurs
    def where_is(self, token, sentence_dict):
        found = {}
        for sid in sentence_dict:
            found[sid] = [i for i, word in enumerate(sentence_dict[sid]) if word == token]
        return found
        
def tokenize_en(model,text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in model.tokenizer(text)]
        
def main():
    model = AttentionMechanism()
    spacy_en = spacy.load('en_core_web_sm')

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    MIN_DOC_LEN = 20
    
    with open('datasets/WikiText2/wikitext-2/wiki.train.tokens','r', encoding='utf-8') as f:
        for line in f.readlines()[:1000]:
            if line.strip() and len(line.split()) >= MIN_DOC_LEN:
                model.train_data.append(line)
        
    with open('datasets/WikiText2/wikitext-2/wiki.valid.tokens','r', encoding='utf-8') as f:
        for line in f.readlines()[:1000]:
            if line.strip() and len(line.split()) >= MIN_DOC_LEN:
                model.val_data.append(line)
                model.val_sentence_ids[len(model.val_data)] = tokenize_en(spacy_en,line)
        
    with open('datasets/WikiText2/wikitext-2/wiki.test.tokens','r', encoding='utf-8') as f:
        for line in f.readlines()[:1000]:
            if line.strip() and len(line.split()) >= MIN_DOC_LEN:
                model.test_data.append(line)
                model.test_sentence_ids[len(model.test_data)] = tokenize_en(spacy_en,line)

    counter = Counter()
    for line in model.train_data:
        counter.update(tokenize_en(spacy_en,line))
    
    vocab = build_vocab_from_iterator([counter.elements()], min_freq=1, specials= ['<unk>', '<pad>', '<mask>'])
    
    model.vocab = vocab
    model.vocab.set_default_index(model.vocab['<unk>'])
    model.text_pipeline = lambda x: [model.vocab[token] for token in tokenize_en(spacy_en,x)]

    print(f"Vocab size: {len(model.vocab)}")
    
    model.build_model()
        
    print(f"The model has {model.count_parameters(model.s2s):,} trainable parameters")
    
    model.s2s.apply(model.initialize_weights)
    
    BATCH_SIZE = 128
    N_EPOCHS = 8
    CLIP = 1

    n_minibatches = ceil(len(model.train_data) / BATCH_SIZE)
    print(f"Number of minibatches: {n_minibatches}")

    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch+1:02}')
        
        start_time = time.time()
        
        for i in range(n_minibatches):
            print(f'\tMinibatch {i}')
            minibatch = model.train_data[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            masked_text, unmasked_text = model.process_batch(minibatch, model.MASK_IDX, model.MASK_PERCENT)
            train_loss = model.train(model, masked_text, unmasked_text, CLIP)
        
        val_src, val_trg = model.process_batch(model.val_data, model.MASK_IDX, model.MASK_PERCENT)
        valid_loss = model.evaluate(model, val_src, val_trg)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = model.epoch_time(start_time, end_time)
        
        if valid_loss < model.best_valid_loss:
            model.best_valid_loss = valid_loss
        
        print(f'\tTime: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):7.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f} |  Val. PPL: {math.exp(valid_loss):7.4f}')

if __name__ == "__main__":
    main()
