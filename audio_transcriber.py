import os
from nemo.collections.asr.models import EncDecCTCModel

# Load ASR model once
def load_asr_model(asr_model_id):
    print("Loading ASR model...")
    model = EncDecCTCModel.from_pretrained(asr_model_id) 
    print("ASR model loaded")
    return model


def transcribe_audio_folder(asr_model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    for filename in os.listdir(input_dir):
        if filename.endswith((".wav", ".mp3")):
            audio_path = os.path.join(input_dir, filename)
            output_file = os.path.join(
                output_dir, filename.replace(".wav", ".txt").replace(".mp3", ".txt")
            )
            print(f"Transcribing: {audio_path}")

            try:
                transcription = asr_model.transcribe([audio_path])[0]
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(transcription)
                print(f"Transcription saved: {output_file}")
            except Exception as e:
                print(f"Error transcribing {audio_path}: {e}")

    print(f"🎯 All transcriptions saved in: {output_dir}")
