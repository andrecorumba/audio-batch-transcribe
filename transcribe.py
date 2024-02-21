import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

import os

import sqlite3


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

# Load opus file in folder 
folder = 'audios'
#folder = '/mnt/d/audios'
files = os.listdir(folder)

# Conexão 
print('iniciando ...')
with sqlite3.connect('database/audios_transcriptions.db') as con:
    cur = con.cursor()
  
    # Create table if not exists
    cur.execute('''CREATE TABLE IF NOT EXISTS transcriptions
                   (id INTEGER PRIMARY KEY, 
                    file TEXT,
                    text TEXT)'''
                )
    
    # Create a table for errors
    cur.execute('''CREATE TABLE IF NOT EXISTS errors
                   (id INTEGER PRIMARY KEY, 
                    file TEXT,
                    error TEXT)'''
                )

    for i, file in enumerate(files):
        if file.endswith('opus'):
            with open(os.path.join(folder,file), 'rb') as f:
                sample = f.read()
                try:
                    result = pipe(sample)
                    print(f'file: {file} number: {i} from {len(files)}')    
                    #print(result["text"])
            
                    # Insert into database
                    cur.execute('INSERT INTO transcriptions (file, text) VALUES (?, ?)', (file, result["text"]))
                    con.commit()
                except Exception as e:
                    print(f'Error: {e}')
                    print(f'file: {file} number: {i} from {len(files)}')    
                    # Insert into error database
                    cur.execute('INSERT INTO errors (file, error) VALUES (?, ?)', (file, str(e)))

print('concluído!')
# Close connection
# con.close()


