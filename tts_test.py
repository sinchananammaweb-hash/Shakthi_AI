import torch
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer
import os

MODEL_ID = "facebook/mms-tts-kan"

print("Loading Kannada TTS model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = VitsModel.from_pretrained(MODEL_ID)

test_sentences = [
    "ನಮಸ್ಕಾರ, ನಾನು ಶಕ್ತಿ AI. ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
    "ರಕ್ತದಲ್ಲಿರುವ ಹಿಮೋಗ್ಲೋಬಿನ್ ದೇಹದ ವಿವಿಧ ಭಾಗಗಳಿಗೆ ಆಮ್ಲಜನಕವನ್ನು ಸಾಗಿಸುತ್ತದೆ.",
    "ನಿಮಗೆ ತುಂಬಾ ಸುಸ್ತು ಅನಿಸುತ್ತಿದ್ದರೆ ಸಾಕಷ್ಟು ವಿಶ್ರಾಂತಿ ಮತ್ತು ನೀರು ಕುಡಿಯುವುದು ಮುಖ್ಯ.",
    "ಐರನ್ ಮಾತ್ರೆ ಸೇವಿಸಿದ ನಂತರ ಮಲ ಕಪ್ಪಾಗುವುದು ಕೆಲವೊಮ್ಮೆ ಸಾಮಾನ್ಯವಾಗಿರಬಹುದು.",
    "ನಿಮ್ಮ ಆರೋಗ್ಯದ ಬಗ್ಗೆ ಹೆಚ್ಚು ಚಿಂತೆ ಇದ್ದರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ.",
]

os.makedirs("tts_outputs", exist_ok=True)

print("\nGenerating Kannada speech samples...\n")

for i, text in enumerate(test_sentences, start=1):

    print(f"Test {i}: {text}")

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    audio = output.squeeze().cpu().numpy()

    filename = f"tts_outputs/test_{i}.wav"

    scipy.io.wavfile.write(
        filename,
        rate=model.config.sampling_rate,
        data=audio
    )

    print(f"Saved: {filename}\n")

print("All Kannada TTS tests completed successfully.")