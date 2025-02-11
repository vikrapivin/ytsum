# imports
import os

from dlAudio import dlAudio
from transcribe import transcribeAudio, Transcript
from summarize import summarizeVid
import gc
import torch
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# args in
# vid url
ytURL = 'https://www.youtube.com/watch?v=uE9FzUzAdg8'

# whspr args
whspr_model = 'medium'
whspr_device = device
whspr_compute_type="int8"
whspr_batched=False
whspr_batch_size = 16

# summary args
summ_tokens = 1000
final_sum_tokens = 1500
repetition_penalty = 1.05
nwords_final_summ = 100

# run

vidTitle, vidID, filename = dlAudio(ytURL)

# pass to whspr to transcribe
print(f'[Transcribe] switching to transcription.')
transcript, transcript_filename = transcribeAudio(filename, whspr_model, device=whspr_device, compute_type=whspr_compute_type, 
    batched=False, batch_size = 16)
gc.collect() # Python thing

torch.cuda.empty_cache()

# make summary
print(f'[Summarize] switching to summarization.')

full_transcript = ''
for snippet in transcript.snippets:
    full_transcript = full_transcript + snippet.text + ' '
full_transcript = full_transcript.strip()

summVid = summarizeVid(full_transcript, vidTitle, device=device)
summVid.summarizeSegments(summ_tokens, repetition_penalty = repetition_penalty)
summVid.fullSummary(nwords_final_summ, final_sum_tokens)
print(summVid.finSum)

# save output
with open(f'summs/{filename[3:-4]}_summary.txt','w') as summ_file:
    summ_file.write(summVid.finSum)
with open(f'summs/{filename[3:-4]}_debug.txt','w') as summ_file:
    summ_file.write(summVid.summ_output)
    summ_file.write('\n\n\n\n\n')
    summ_file.write(repr(summVid.summs))