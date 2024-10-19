import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment

tokenizer = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

r = sr.Recognizer()
sampling_rate = 16000

break_keyword = ['bye','quit','close','exit']

with sr.Microphone(sample_rate = sampling_rate) as source: 
    print('You can start speaking now...')
    open = True

    while open:
        audio = r.listen(source) # pyaudio object
        data  = io.BytesIO(audio.get_wav_data()) # list of bytes
        clip  = AudioSegment.from_file(data) # numpy array
        x     = torch.FloatTensor(clip.get_array_of_samples()) # tensor

        inputs = tokenizer(x, sampling_rate = sampling_rate, return_tensors = 'pt', padding = 'longest').input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis=-1)
        text_list   = tokenizer.batch_decode(tokens) # tokens to string
        text_str   = str(text_list).lower()
        print('You said: ', text_str)
        if text_list[0].lower() in break_keyword:
            print('Exiting program...')
            break