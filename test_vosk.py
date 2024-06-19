import wave
import json
import logging
from vosk import Model, KaldiRecognizer
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def segment_audio(file_path, segment_duration=10):
    with wave.open(file_path, 'rb') as wf:
        segments = []
        frame_rate = wf.getframerate()
        num_frames = wf.getnframes()
        total_duration = num_frames / frame_rate
        for start in range(0, int(total_duration), segment_duration):
            end = min(start + segment_duration, total_duration)
            wf.setpos(start * frame_rate)
            frames = wf.readframes(int((end - start) * frame_rate))
            segments.append(frames)
    return segments, frame_rate

def recognize_segment(args):
    segment, model_path, frame_rate = args
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, frame_rate)
    recognizer.AcceptWaveform(segment)
    return recognizer.Result()

def recognize_audio(model_path, segments, frame_rate):
    args = [(segment, model_path, frame_rate) for segment in segments]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(recognize_segment, args), total=len(segments), desc="Processing segments"))
    return results

def align_text_with_audio(text, results):
    words = text.split()
    word_index = 0
    aligned_results = []

    for result in results:
        result_json = json.loads(result)
        if 'result' in result_json:
            for word_info in result_json['result']:
                word = word_info
                word = word_info['word']
                while word_index < len(words) and words[word_index] != word:
                    word_index += 1
                if word_index < len(words):
                    aligned_results.append({
                        "word": words[word_index],
                        "start": word_info['start'],
                        "end": word_info['end']
                    })
                    word_index += 1
    return aligned_results

def save_to_json(aligned_results, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(aligned_results, file, ensure_ascii=False, indent=4)

def main():
    logging.info("Reading text file.")
    text_file = "./test_Vocals_text.txt"
    audio_file = "./test_Vocals.wav"
    model_path = "./vosk-model-ru-0.42"

    text = read_text_file(text_file)
    logging.info("Text file read successfully.")

    logging.info("Segmenting audio file.")
    segments, frame_rate = segment_audio(audio_file)
    logging.info("Audio file segmented successfully.")

    logging.info("Recognizing audio segments.")
    results = recognize_audio(model_path, segments, frame_rate)
    logging.info("Audio recognition completed.")

    logging.info("Aligning text with audio.")
    aligned_results = align_text_with_audio(text, results)
    logging.info("Text aligned with audio successfully.")

    logging.info("Saving results to JSON.")
    save_to_json(aligned_results, "output.json")
    logging.info("Results saved to output.json.")

if __name__ == "__main__":
    main()
