from transformers import AutoTokenizer, AutoModelForCausalLM

def word_count(text):
    return len(text.split())
def segmentText(text, word_count = 400, start_overlap=50, end_overlap=50):
    '''Segment text into word_count segments, with overlap words from previous segments'''
    '''Returns an array of arrays'''
    '''The array within the arrays gives the context length at the start, at the end, and then text the segment with '''
    '''the first segment of word_count + overlap length, later segments with word_count+2*overlap length'''
    '''last segment has less than word_count + overlap length'''

    all_words = text.split()

    tot_wrds = len(all_words)
    total_segments = int(tot_wrds/word_count)
    if tot_wrds % word_count != 0: # add in one segment unless perfectly divided by the word count
        total_segments = total_segments + 1

    segmented_text = []
    for ii in range(0, total_segments):
        # where to start
        start_context = 0
        end_context = 0
        start_loc = ii*word_count
        if ii == 0:
            pass
        else:
            start_loc -= start_overlap
            start_context = start_overlap
        end_loc = (ii+1)*word_count + end_overlap
        if end_loc > tot_wrds:
            end_loc = tot_wrds
        else:
            end_context = end_overlap
        segmented_text.append([start_context, end_context, ' '.join(all_words[start_loc:end_loc])])
    return segmented_text

class summarizeVid:
    def __init__(self, transcript, vidTitle, word_count = 400, context_length_start=50, context_length_end=50, device='cpu'):
        """Initialize the summary class with the transcript, and some other parameters
        TODO: documentation"""
        self.transcript = transcript
        self.context_length_start = context_length_start
        self.context_length_end = context_length_end
        self.vidTitle = vidTitle
        self.device = device
        self.segmentPrompt = "Only output the keys points of each segment, and while you can use the contexual words for reference, do not include information from the contextual words in your keys points. For reference, the title of the video is \"{self.vidTitle}\". If the segment contains any information relevant to the title of the video, include that information in the key points as well. Do not use any informal terminology."
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").to(self.device)
        self.segmented_text = segmentText(self.transcript, word_count= word_count, start_overlap=context_length_start, end_overlap=context_length_end)
    def summarizeSegments(self, max_new_tokens, **kwargs):
        self.summs = []
        segment_count = 1
        for segm in self.segmented_text:
            print(f'Summarizing segment {segment_count}.')
            context_start = int(segm[0])
            context_end = int(segm[1])
            segm_text = segm[2]
            summ = self.summarizeSegment(segm_text, context_start, context_end, max_new_tokens = max_new_tokens, **kwargs)
            self.summs.append(summ)
            segment_count += 1
    
    def summarizeSegment(self, sumText, context_length_start, context_length_end, **kwargs):
        """Summarize segments from video
        TODO: documentation"""
        nwords_summary = int(0.2*word_count(sumText))
        prompt =  f"Your goal is to summarize the given segment of a YouTube video transcript in maximum {nwords_summary} words. For context, the segment includes an additional {context_length_start} words in the beginning and {context_length_end} words at the end. {self.segmentPrompt}"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": sumText}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        attention_mask = model_inputs['attention_mask']
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask, **kwargs
        )
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        return response
    def summarizeSummariesFull(self, sumText, vidTitle, nwords_summary, **kwargs):
        prompt =  f"Your goal is to create a summary in under {nwords_summary} words from a list of summaries of different segments of a YouTube video. Your goal is for the reader to be able to understand what happened without any additional context. The title of the video is: \"{vidTitle}\". If the title has a clickbait question in it, answer the clickbait question. Do not use any informal terminology, do not repeat the title, do not use terms like author."

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": sumText}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        attention_mask = model_inputs['attention_mask']
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask, **kwargs
        )
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        return response
    def fullSummary(self, nwords_summary, max_new_tokens):
        if len(self.summs) == 0:
            print('Summarize segments first by running summarizeSegments().')
            # TODO: implement error raising here
            return -1
        self.summ_output = ''
        part_count = 1
        for ii in self.summs:
            summ = ii[0]
            output = summ.split('</think>\n\n')[1]
            eos_token = self.tokenizer.decode(self.tokenizer.eos_token_id)
            if eos_token not in output:
                print(f"Error in summarizing, ran out of new words in part {part_count}")
            else:
                output = output[0:-len(eos_token)] # remove end of sentence
            self.summ_output = self.summ_output + f'{part_count}. ' + output + '\n'
            part_count = part_count + 1
        finSum = self.summarizeSummariesFull(self.summ_output, self.vidTitle, nwords_summary, max_new_tokens = max_new_tokens)
        self.finSum = finSum[0]




# todo: fix CUDA send
