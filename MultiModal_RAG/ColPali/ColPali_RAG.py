import os
import numpy as np
import torch
import openai
import argparse
import time
import speech_recognition as sr

from moviepy.editor import VideoFileClip
from tqdm import tqdm
from typing import List, cast
from dotenv import load_dotenv

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device

from scenedetect import open_video,SceneManager,StatsManager, save_images
from scenedetect.detectors import ContentDetector


from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

def video_to_images(video_path, output_folder):
    output = output_folder
    video = open_video(video_path)

    scene_manager = SceneManager(stats_manager=StatsManager())
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()
    for index, scene in enumerate(scene_list):
        padded_index = f'{index:03}'
        save_images(scene_list=[scene], 
                    video=video,
                    image_extension='png',
                    image_name_template=f'$VIDEO_NAME-Scene-{padded_index}',
                    output_dir=output,
                    num_images=1)

def video_to_audio(video_path, output_audio_path):
    
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)
    
def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)

    with audio as source:
        # Record the audio data
        audio_data = recognizer.record(source)

        try:
            # Recognize the speech
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from service; {e}")

    return text
    
def retrieve(output_folder,query):
    start_time = time.time()
    Embedding_model_name = "vidore/colpali-v1.2"
    Embedding_model = ColPali.from_pretrained(
        Embedding_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
    ).eval()

    processor = ColPaliProcessor.from_pretrained(Embedding_model_name)    
    # フォルダ内のPNGファイルをファイル名順に取得
    images = []
    png_files = sorted([filename for filename in os.listdir(output_folder) if filename.endswith('.png')])
    
    # 画像を開いてリストに追加
    for filename in png_files:
        image_path = os.path.join(output_folder, filename)
        images.append(Image.open(image_path))

    # Run inference - docs
    dataloader = DataLoader(
        dataset=ListDataset[str](images),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    ds: List[torch.Tensor] = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(Embedding_model.device) for k, v in batch_doc.items()}
            embeddings_doc = Embedding_model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    # Run inference - queries
    dataloader = DataLoader(
        dataset=ListDataset[str]([query]),
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )
    
    qs: List[torch.Tensor] = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(Embedding_model.device) for k, v in batch_query.items()}
            embeddings_query = Embedding_model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
    scores = processor.score(qs, ds).cpu().numpy()
    idx_top_n = scores.argsort(axis=1)[:, -5:][:, ::-1]
    end_time = time.time()

    # 実行時間を表示
    execution_time = end_time - start_time
    print("********** Retrieve Time **********\n")
    print(f"Execution time: {execution_time} seconds\n")
    return idx_top_n
    
def run_llama(txt, query, output_folder, qa_tmpl_str, idx_top_n):
    
    # 入力画像
    img = []
    for i in idx_top_n[0]:
        img.append(f"{output_folder}input_vid-Scene-{str(i).zfill(3)}.png")

    # クエリ
    query_str = query

    # ドキュメント
    image_documents = SimpleDirectoryReader(
        input_dir=output_folder, input_files=img
    ).load_data()
    context_str = "".join(txt)
    # LLM読み込み
    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'), max_new_tokens=1500
    )

    # 回答文を生成
    response_1 = openai_mm_llm.complete(
        prompt=qa_tmpl_str.format(
            context_str=context_str, query_str=query_str, ),
        image_documents=image_documents,
    )
    print("************** QUERY **************\n")
    print(query)
    print("\n************** OUTPUT **************\n")
    print(response_1.text)
    print("\n************************************")
    
def generate_answer(txt, output_folder, query, idx_top_n):
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    # テキスト情報をまとめる
    
    qa_tmpl_str = (
        """
     Given the provided information, including relevant images and retrieved context from the video, \
     accurately and precisely answer the query without any additional prior knowledge.\n"
        "Please ensure honesty and responsibility, refraining from any racist or sexist remarks.\n"
        "---------------------\n"
        "Context: {context_str}\n"
        "---------------------\n"
        "Query: {query_str}\n"
        "Answer: "
    """
    )
    run_llama(txt, query, output_folder, qa_tmpl_str, idx_top_n)
    
def main(filename,query):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    video_path = './video/' + filename
    output_folder = "./img/" + filename + '/'
    output_audio_path = output_folder + "output_audio.wav"

    video_to_images(video_path, output_folder)
    video_to_audio(video_path, output_audio_path)
    txt = audio_to_text(output_audio_path)
    idx_top_n = retrieve(output_folder, query)
    generate_answer(txt, output_folder, query, idx_top_n)
    

if __name__ == "__main__":
    # argparseを使ってコマンドライン引数を設定
    parser = argparse.ArgumentParser(description="Process a file name from the command line.")
    
    # ファイル名を引数として追加
    parser.add_argument('filename', type=str, help="The name of the file to process")
    query = input("What would you like to know about this video?:")
    
    # 引数をパース
    args = parser.parse_args()

    # main関数に引数としてファイル名を渡す
    main(args.filename,query)
    