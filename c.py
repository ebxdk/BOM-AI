import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf

# Load the processor and model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load xvector speaker embeddings
speaker_embeddings = torch.load("https://huggingface.co/microsoft/speecht5_tts/blob/main/embeddings/speaker_embeddings.pt")

# Select a speaker embedding (e.g., the first one)
selected_speaker = speaker_embeddings[0].unsqueeze(0)

# Text to be converted to speech
text = "Hello, how can I assist you today?"

# Preprocess text
inputs = processor(text=text, return_tensors="pt")

# Generate speech
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=selected_speaker, vocoder=vocoder)

# Save the speech to a file
sf.write("output.wav", speech.numpy(), samplerate=16000)
