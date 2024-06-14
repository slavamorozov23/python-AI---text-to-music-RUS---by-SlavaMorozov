import os
import re
import librosa
import numpy as np
from joblib import Memory
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
import h5py
from concurrent.futures import ThreadPoolExecutor

# Настройка кэширования
memory = Memory("./cachedir", verbose=0)

# Функция для очистки текстов
@memory.cache
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Функция для сегментирования аудио
@memory.cache
def segment_audio(file_path, segment_length=5):
    y, sr = librosa.load(file_path)
    segment_samples = segment_length * sr
    segments = [y[i:i+segment_samples] for i in range(0, len(y), segment_samples)]
    return segments, sr

# Функция для получения длительности аудио
@memory.cache
def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000  # Длительность в секундах

# Функция для синхронизации текста и аудио с использованием Vosk
@memory.cache
def synchronize_text_audio(text, audio_file):
    model = Model("./vosk-model-ru-0.42")
    recognizer = KaldiRecognizer(model, 16000)
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    results = []
    for chunk in range(0, len(audio), 4000):
        segment = audio[chunk:chunk + 4000]
        recognizer.AcceptWaveform(segment.raw_data)
        result = recognizer.Result()
        results.append(result)
    
    recognized_text = " ".join([res["text"] for res in results])
    text_lines = text.split('\n')
    
    # Используем временные метки для синхронизации
    timestamps = []
    for line in text_lines:
        words = line.split()
        for word in words:
            if word in recognized_text:
                timestamps.append(recognized_text.find(word))
            else:
                timestamps.append(-1)
    
    return list(zip(text_lines, timestamps))

# Функция для сохранения сегментов аудио и текста
def save_segments(segments, texts, filename='data.h5'):
    with h5py.File(filename, 'w') as f:
        for i, (segment, text) in enumerate(zip(segments, texts)):
            grp = f.create_group(str(i))
            grp.create_dataset('audio', data=segment, compression='gzip')
            grp.create_dataset('text', data=text)

# Функция для параллельного сохранения данных
def parallel_save_segments(segments_texts_pairs, filename='data.h5'):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_segments, seg, txt, filename) for seg, txt in segments_texts_pairs]
        for future in futures:
            future.result()

# Основная функция для выполнения всех шагов предобработки
def preprocess_songs(directory, segment_length=5):
    files = [f for f in os.listdir(directory) if f.endswith('.mp3')]
    for file in files:
        base_name = os.path.splitext(file)[0]
        text_file = os.path.join(directory, base_name + '.txt')
        audio_file = os.path.join(directory, file)
        
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        cleaned_text = clean_text(text)
        audio_segments, sr = segment_audio(audio_file, segment_length)
        segments_texts_pairs = [(audio_segments, synchronize_text_audio(cleaned_text, audio_file))]
        
        parallel_save_segments(segments_texts_pairs, f'{base_name}.h5')

# Пример использования
if __name__ == "__main__":
    directory = "./music_with_text"  # Директория с аудио и текстовыми файлами
    preprocess_songs(directory)
