import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import array
import zlib
import argparse
import os

class GPTZip:
    def __init__(self):
        self.CONTEXT_SIZE = 256
        self.BATCH_SIZE = 10
        self.model = AutoModelForCausalLM.from_pretrained("./code/TinyLlama", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("./code/TinyLlama", trust_remote_code=True)
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu" 
        self.device = torch.device(dev) 
        self.model.to(self.device) 


    def text_to_tokens(self, text):
        # ignore the warning that this gives about too many tokens
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens = tokens["input_ids"].squeeze()
        return tokens
    
    def tokens_to_text(self, tokens):
        tokens = tokens.reshape((1, -1))
        text = self.tokenizer.batch_decode(tokens)
        return text[0]

    def pad(self, tokens, padding_val):
        pad_len = self.CONTEXT_SIZE - tokens.shape[0] % self.CONTEXT_SIZE
        if pad_len != self.CONTEXT_SIZE:
            padding = torch.tensor([padding_val]*pad_len)

            tokens = torch.cat((tokens, padding))

        else:
            pad_len = 0
  
        
        return tokens, pad_len
    
    @torch.no_grad()
    def get_logits(self, tokens, token_index, past=None):
        input_ids = tokens[:, token_index].reshape(-1, 1)
        output = self.model(input_ids=input_ids, past_key_values=past)
        logits = output.logits 
        if len(logits.shape) > 2:
            logits = logits.reshape((logits.shape[0], -1))
        return logits, output.past_key_values
    

    def encode_one_batch(self, tokens, token_index, past=None):

        assert len(tokens.shape) == 2
 
        logits, past = self.get_logits(tokens, token_index, past)
        assert len(logits.shape) == 2
        logits, sorted_tokens = torch.sort(logits, descending=True)
        
        assert len(sorted_tokens.shape) == 2


        next_tokens = tokens[:, token_index + 1]
        next_tokens_expanded = next_tokens.view(-1, 1).expand_as(sorted_tokens)
        next_tokens_expanded = next_tokens_expanded

        # Find score as index of next tokens
        scores = (sorted_tokens==next_tokens_expanded).nonzero(as_tuple=True)

        scores = scores[1] # remove index column

        return scores, past

    def decode_one_batch(self, input_tokens, scores, score_index, past=None):
        assert len(scores.shape) == 2
        logits, past = self.get_logits(input_tokens, score_index, past)

        logits, sorted_tokens = torch.sort(logits, descending=True)
        assert len(sorted_tokens.shape) == 2
        # the scores give the indexes of the decoded tokens
        indexes = scores[:, score_index].flatten()
        decoded_tokens = sorted_tokens[torch.arange(indexes.shape[0]), indexes]

        return decoded_tokens.int(), past


    def encode(self, text):
        tokens = self.text_to_tokens(text)
        return self.encode_tokens(tokens)

    def encode_tokens(self, tokens):

        tokens, pad_len = self.pad(tokens, self.tokenizer.eos_token_id)
        tokens = tokens.view(-1, self.CONTEXT_SIZE)

        output_scores = torch.zeros(tokens.shape)


        # Add eos to the start of each block (to give it somewhere to start)
        eos = torch.tensor([self.tokenizer.eos_token_id]*tokens.shape[0]).unsqueeze(1)
        tokens = torch.cat((eos, tokens), 1)

        tokens = tokens.to(self.device)

        batches = tokens.shape[0]//self.BATCH_SIZE
        if tokens.shape[0] % self.BATCH_SIZE != 0:
            batches += 1

        # score each batch
        print("Encoding")
        for i in range(batches):
            cur_tokens = tokens[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE]
            cur_output_scores = torch.zeros((cur_tokens.shape[0], cur_tokens.shape[1]-1))
            past = None
            print(i, "out of", batches)
            
            for j in range(cur_tokens.shape[1]-1):

                cur_output_scores[:, j], past = self.encode_one_batch(cur_tokens, j, past)
            output_scores[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE] = cur_output_scores

        output_scores = output_scores.flatten().int()
        if pad_len > 0:
            output_scores = output_scores[:-pad_len]
        return output_scores
    
    def decode(self, scores):
        output_tokens = self.decode_tokens(scores)
        text = self.tokenizer.batch_decode(output_tokens)
        text = "".join(text)
        #text = text.replace("<|endoftext|>", "")
        text = text.replace("<s>", "").replace("</s>", "")
        return text
    
    def decode_tokens(self, scores):

        scores, pad_len = self.pad(scores, self.tokenizer.eos_token_id)

        scores = scores.view(-1, self.CONTEXT_SIZE) # all rows, CONTEXT_SIZE

        output_tokens = torch.zeros(scores.shape, dtype=int)


        # Add eos to the start of each block (to give it somewhere to start)
        eos = torch.tensor([self.tokenizer.eos_token_id]*output_tokens.shape[0]).unsqueeze(1)
        output_tokens = torch.cat((eos, output_tokens), 1) # all rows, CONTEXT_SIZE + 1

        output_tokens = output_tokens.to(self.device)

        batches = scores.shape[0]//self.BATCH_SIZE
        if scores.shape[0] % self.BATCH_SIZE != 0:
            batches += 1

        # score each batch
        print("Decoding")
        for i in range(batches):
            print(i, "out of", batches)
            cur_scores = scores[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE] # BATCH_SIZE, CONTEXT_SIZE

            cur_output_tokens = output_tokens[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE] # BATCH_SIZE, CONTEXT_SIZE
            cur_output_tokens = cur_output_tokens.to(self.device)
            past = None
            for j in range(scores.shape[1]):
 
                cur_output_tokens[:, j+1], past = self.decode_one_batch(cur_output_tokens, cur_scores, j, past) # BATCH_SIZE

            output_tokens[i*self.BATCH_SIZE:(i + 1)*self.BATCH_SIZE] = cur_output_tokens
        
  

        output_tokens = output_tokens[:, 1:].int()
        output_tokens = output_tokens.flatten()

        if pad_len != 0:
            output_tokens = output_tokens[:-pad_len]

        return output_tokens



    def encode_and_zip(self, text):
        encoded = self.encode(text)
        encoded = array.array("H", encoded)
        return zlib.compress(encoded, level=9)
    

    def unzip_and_decode(self, zipped):
        unzipped = zlib.decompress(zipped)
        unzipped = array.array("H", unzipped)
        decoded = self.decode(torch.tensor(unzipped))
        return decoded
    
    def zip_file(self, text_file, zip_file):
        with open(text_file, encoding="utf-8") as f:
            text = f.read()

        zipped = self.encode_and_zip(text)

        with open(zip_file, "wb") as f:
            f.write(zipped)

    def unzip_file(self, zip_file, text_file):
        with open(zip_file, "rb") as f:
            zipped = f.read()
        text = self.unzip_and_decode(zipped)

        with open(text_file, "w", encoding="utf-8") as f:
            f.write(text)

        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                        prog='TinyZip',
                        description='Zips files using LLMzip algorithm with TinyLlama')
    parser.add_argument('-z', '--zip')
    parser.add_argument('-u', '--unzip')
    parser.add_argument('-o', '--output_file')
    args = parser.parse_args()   

    if args.zip is not None:
        # 修复：判断是否为 .txt 文件（通用写法）
        if not args.zip.endswith(".txt"):
            print("This program can only zip text files")
            exit()
        print("Loading TinyLlama model...")
        gpt_zip = GPTZip()
    
        # 修复：安全去掉 .txt 后缀，再加上 .tinz
        zip_file = os.path.splitext(args.zip)[0] + ".tinz"
    
        if args.output_file is not None:
            zip_file = args.output_file
            # 修复：安全判断后缀
            if not zip_file.endswith(".tinz"):
                zip_file += ".tinz"
    
        gpt_zip.zip_file(args.zip, zip_file)
        print("Saved to: ", zip_file)

    elif args.unzip is not None:
        # 已经是正确的，保留
        if not args.unzip.endswith(".tinz"):
            print("This program can only unzip TinyZip files (.tinz)")
            exit()
        print("Loading TinyLlama model...")
        gpt_zip = GPTZip()
    
        text_file = os.path.splitext(args.unzip)[0] + ".txt"
    
        if args.output_file is not None:
            text_file = args.output_file
            # 修复：安全判断后缀
            if not text_file.endswith(".txt"):
                text_file += ".txt"
    
        gpt_zip.unzip_file(args.unzip, text_file)
        print("Saved to: ", text_file)

    else:
        print("Usage: TinyZip -z input.txt    (compress)")
        print("       TinyZip -u input.tinz    (decompress)")