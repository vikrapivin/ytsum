from faster_whisper import WhisperModel, BatchedInferencePipeline
import os

# quick class to manage transcription
class Transcript:
    class Snippet:
        def __init__(self, start_time, end_time, text):
            """Initialize a Snippet with a start time, end time, and the transcript text"""
            self.start_time = start_time
            self.end_time = end_time
            self.text = text

        def __repr__(self):
            """Represent the Snippet as a string for easier viewing"""
            return f"Snippet({self.start_time}, {self.end_time}, {repr(self.text)})"
        def to_string(self):
            """Represent the Snippet as a string for easier viewing"""
            return f"{self.start_time}, {self.end_time}, {self.text}"

        @classmethod
        def from_string(cls, snippet_str):
            """Create a Snippet object from a string representation"""
            parts = snippet_str.split(', ', 2)
            start_time = float(parts[0])
            end_time = float(parts[1])
            text = parts[2].strip()
            return cls(start_time, end_time, text)

    def __init__(self):
        """Initialize the Transcript with an empty list of snippets"""
        self.snippets = []

    def add_snippet(self, start_time, end_time, text):
        """Add a new Snippet to the Transcript"""
        snippet = self.Snippet(start_time, end_time, text)
        self.snippets.append(snippet)

    def sort_snippets_by_start_time(self):
        """Sort the snippets by start time in ascending order"""
        self.snippets.sort(key=lambda snippet: snippet.start_time)

    def __repr__(self):
        """Represent the Transcript with all snippets"""
        return f"Transcript({repr(self.snippets)})"
    
    def __iter__(self):
        """Generator to iterate over all snippets"""
        for snippet in self.snippets:
            yield snippet

    def save_to_file(self, filename):
        """Save the Transcript to a text file"""
        with open(filename, 'w') as file:
            for snippet in self.snippets:
                file.write(snippet.to_string() + '\n')

    def load_from_file(self, filename):
        """Load the Transcript from a text file"""
        if not os.path.exists(filename):
            print(f"File {filename} does not exist.")
            return

        self.snippets = []
        with open(filename, 'r') as file:
            for line in file:
                snippet = self.Snippet.from_string(line.strip())
                self.snippets.append(snippet)

def transcribeAudio(file_name, model_size, save_name="", device='cpu', compute_type="int8", batched=False, batch_size = 16, **kwargs):
    """
    transcribeAudio - transcribe audio file using whspr model
    
    Arguments:
        file_name: File to save file as
        model_size: Which whspr model to use
        save_name: leave blank for default, save file path
        batch: use batch processing
        batch_size: refer to whspr documentation
        
    returns:
     - save_name- returns the saved transcript name"""
    transcript = Transcript()
    if save_name == "":
        save_name = f"{file_name[0:-4]}.txt"
    if os.path.isfile(save_name):
        transcript.load_from_file(save_name)
        return transcript, save_name
    if batched: # not tested extensively yet
        model = WhisperModel(model_size, device=device, compute_type=compute_type, **kwargs)
        batched_model = BatchedInferencePipeline(model=model) 
        segments, info = batched_model.transcribe(file_name, batch_size=batch_size)
    else:
        model = WhisperModel(model_size, device=device, compute_type=compute_type, **kwargs)
        segments, info = model.transcribe(file_name, beam_size=5)
    
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        transcript.add_snippet(segment.start, segment.end, segment.text)
    
    transcript.sort_snippets_by_start_time()
    transcript.save_to_file(save_name)

    return transcript, save_name
