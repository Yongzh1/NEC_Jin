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

from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext

from llama_index.vector_stores.lancedb import LanceDBVectorStore
import openai
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode

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

def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text

def main(filename,query):
    video_path = './video/' + filename
    output_folder = "./img/" + filename + '/'
    output_audio_path = output_folder + "output_audio.wav"
    load_dotenv()

    video_to_images(video_path, output_folder)
    video_to_audio(video_path, output_audio_path)
    text_data = audio_to_text(output_audio_path)

    # data processing
    
    with open(output_folder + "output_text.txt", "w") as file:
        file.write(text_data)
    print("Text data saved to file")
    file.close()
    os.remove(output_audio_path)
    print("Audio file removed")


    # Building Vector Store

    start_time = time.time()
    
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
    image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )
    documents = SimpleDirectoryReader(output_folder).load_data()
    
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    
    # Retrieving Relavant Images and Context
    retriever_engine = index.as_retriever(
        similarity_top_k=5, image_similarity_top_k=5
    )
    query_str = query
    img, txt = retrieve(retriever_engine=retriever_engine, query_str=query_str)
    image_documents = SimpleDirectoryReader(
        input_dir=output_folder, input_files=img
    ).load_data()
    context_str = "".join(txt)

    end_time = time.time()
    
    execution_time = end_time - start_time
    print("************** Retrieve Time **************\n")
    print(f"Execution time: {execution_time} seconds")
    print("\n*****************************************")
    
    #Generation 
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
    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'), max_new_tokens=1500
    )
    
    
    response_1 = openai_mm_llm.complete(
        prompt=qa_tmpl_str.format(
            context_str=context_str, query_str=query_str, 
        ),
        image_documents=image_documents,
    )
    
    print(response_1.text)

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
    