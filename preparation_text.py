import wave
import json
import logging
from vosk import Model, KaldiRecognizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Шаг 1: Установка модели и подготовка аудиофайла
model_path = "./vosk-model-ru-0.42"
wav_path = "./test_Vocals.wav"
txt_path = "./test_Vocals_text.txt"
output_json_path = "./output_timestamps.json"



# Convert stereo to mono using pydub
logging.info("Opening and converting WAV file...")
audio = AudioSegment.from_wav(wav_path)
if audio.channels != 1:
    logging.info("Converting stereo to mono...")
    audio = audio.set_channels(1)
    audio.export(wav_path, format="wav")

# Open the converted wav file
wf = wave.open(wav_path, "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 44100, 48000]:
    raise ValueError("Audio file must be WAV format mono PCM.")

def recognize_chunk(chunk, recognizer, framerate):
    recognizer.AcceptWaveform(chunk)
    return recognizer.Result()

# Шаг 2: Распознавание речи с использованием многопоточности
logging.info("Starting speech recognition...")
chunk_size = 400
total_frames = wf.getnframes()
audio_chunks = [wf.readframes(chunk_size) for _ in range(total_frames // chunk_size)]

logging.info("Loading model...")
model = Model(model_path)

# Use a thread pool for concurrent processing
results = []
with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the number of workers as needed
    futures = [executor.submit(recognize_chunk, chunk, KaldiRecognizer(model, wf.getframerate()), wf.getframerate()) for chunk in audio_chunks]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing audio"):
        results.append(future.result())

# Add the final result
final_recognizer = KaldiRecognizer(model, wf.getframerate())
results.append(final_recognizer.FinalResult())

# Шаг 3: Парсинг результатов
logging.info("Parsing results...")
words = []
for result in results:
    result_dict = json.loads(result)
    if 'result' in result_dict:
        words.extend(result_dict['result'])

# Шаг 4: Загрузка текста песни
logging.info("Loading lyrics...")
with open(txt_path, "r", encoding='utf-8') as f:
    lyrics = f.readlines()

# Шаг 5: Соотнесение строк текста с временными метками
logging.info("Matching lyrics with timestamps...")
timestamps = []
current_word_index = 0

for line in tqdm(lyrics, desc="Processing lyrics"):
    line = line.strip()
    if not line:
        continue

    line_words = line.split()
    start_time = None
    end_time = None

    for word in line_words:
        while current_word_index < len(words) and words[current_word_index]['word'] != word:
            current_word_index += 1
        if current_word_index < len(words) and words[current_word_index]['word'] == word:
            if start_time is None:
                start_time = words[current_word_index]['start']
            end_time = words[current_word_index]['end']
            current_word_index += 1

    if start_time is not None and end_time is not None:
        timestamps.append({
            "line": line,
            "start_time": start_time,
            "end_time": end_time
        })

# Шаг 6: Сохранение результатов в JSON
logging.info(f"Saving results to {output_json_path}...")
with open(output_json_path, "w", encoding='utf-8') as f:
    json.dump(timestamps, f, ensure_ascii=False, indent=4)

logging.info(f"Timestamps saved to {output_json_path}")
