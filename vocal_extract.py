import os
import librosa
import numpy as np
import scipy
from pydub import AudioSegment
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_music(file_path, output_path, progress):
    logging.info(f'Start processing {file_path}')
    y, sr = librosa.load(file_path, sr=None)
    progress.update(10)  # Обновление прогресса
    logging.info(f'Loaded file {file_path}')
    
    stft = librosa.stft(y)
    progress.update(10)  # Обновление прогресса
    logging.info(f'Computed STFT for {file_path}')
    
    magnitude, phase = librosa.magphase(stft)
    progress.update(10)  # Обновление прогресса
    logging.info(f'Computed magnitude and phase for {file_path}')
    
    mask = librosa.decompose.nn_filter(magnitude, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sr)))
    progress.update(10)  # Обновление прогресса
    logging.info(f'Computed mask for {file_path}')
    
    filter_magnitude = np.minimum(magnitude, mask)
    progress.update(10)  # Обновление прогресса
    logging.info(f'Applied mask for {file_path}')
    
    filtered_stft = filter_magnitude * phase
    y_filtered = librosa.istft(filtered_stft)
    progress.update(20)  # Обновление прогресса
    logging.info(f'Computed inverse STFT for {file_path}')
    
    scipy.io.wavfile.write(output_path, sr, y_filtered.astype(np.float32))
    progress.update(20)  # Обновление прогресса
    logging.info(f'Successfully wrote output file {output_path}')

def convert_mp3_to_wav(mp3_file, wav_file, progress):
    logging.info(f'Converting {mp3_file} to {wav_file}')
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")
    progress.update(50)  # Обновление прогресса
    logging.info(f'Successfully converted {mp3_file} to {wav_file}')

def process_file(mp3_path, output_directory):
    with tqdm(total=100, desc=os.path.basename(mp3_path)) as progress:
        logging.info(f'Start processing file {mp3_path}')
        filename = os.path.basename(mp3_path)
        wav_path = os.path.join(output_directory, filename.replace('.mp3', '.wav'))
        output_path = os.path.join(output_directory, 'vocal_' + filename.replace('.mp3', '.wav'))
        
        convert_mp3_to_wav(mp3_path, wav_path, progress)
        remove_music(wav_path, output_path, progress)
        os.remove(wav_path)
        progress.update(10)  # Обновление прогресса на завершающем шаге
        logging.info(f'Finished processing file {mp3_path}')

def process_file_wrapper(args):
    process_file(*args)

def process_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    mp3_files = [os.path.join(input_directory, filename) for filename in os.listdir(input_directory) if filename.endswith('.mp3')]
    
    num_workers = int(cpu_count() * 0.8)

    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_file_wrapper, [(mp3_file, output_directory) for mp3_file in mp3_files]), total=len(mp3_files)))

if __name__ == '__main__':
    input_directory = './music_with_text'
    output_directory = './vocal_only'
    process_directory(input_directory, output_directory)
