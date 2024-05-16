import os
import shutil
import random
import string
import numpy as np
import argparse
import pandas as pd
from PIL import Image, ImageFile
from scipy.fftpack import dct
from pathlib import Path
from imgutils.validate import anime_classify_score, anime_real_score,anime_style_age_score,get_monochrome_score
from imgutils.generic import classify_predict_score
from functools import lru_cache
from huggingface_hub import HfFileSystem, hf_hub_download
from natsort import natsorted
from concurrent.futures import ProcessPoolExecutor, as_completed

Image.MAX_IMAGE_PIXELS = None  # This removes the limit on the image size
ImageFile.LOAD_TRUNCATED_IMAGES = True  # This allows to load truncated images
hf_fs = HfFileSystem()
_REPOSITORY = 'deepghs/anime_aesthetic'
_DEFAULT_MODEL = 'swinv2pv3_v0_448_ls0.2_x'
_MODELS = natsorted([
    os.path.dirname(os.path.relpath(file, _REPOSITORY))
    for file in hf_fs.glob(f'{_REPOSITORY}/*/model.onnx')
])

LABELS = ["worst", "low", "normal", "good", "great", "best", "masterpiece"]
@lru_cache()
def _get_mark_table(model):
    df = pd.read_csv(hf_hub_download(
        repo_id=_REPOSITORY,
        repo_type='model',
        filename=f'{model}/samples.csv',
    ))
    df = df.sort_values(['score'])
    df['cnt'] = list(range(len(df)))
    df['final_score'] = df['cnt'] / len(df)
    x = np.concatenate([[0.0], df['score'], [6.0]])
    y = np.concatenate([[0.0], df['final_score'], [1.0]])
    return x, y

def _get_percentile(x, y, v):
    idx = np.searchsorted(x, np.clip(v, a_min=0.0, a_max=6.0))
    if idx < x.shape[0] - 1:
        x0, y0 = x[idx], y[idx]
        x1, y1 = x[idx + 1], y[idx + 1]
        return np.clip((v - x0) / (x1 - x0) * (y1 - y0) + y0, a_min=0.0, a_max=1.0)
    else:
        return y[idx]

def _fn_predict(image, model):
    scores = classify_predict_score(
        image=image,
        repo_id=_REPOSITORY,
        model_name=model,
    )
    weighted_mean = sum(i * scores[label] for i, label in enumerate(LABELS))
    x, y = _get_mark_table(model)
    percentile = _get_percentile(x, y, weighted_mean)
    return weighted_mean, percentile, scores

class PHash:
    def __init__(self):
        self.target_size = (32, 32)
        self.__coefficient_extract = (8, 8)

    def _array_to_hash(self, hash_mat):
        return ''.join('%0.2x' % x for x in np.packbits(hash_mat))

    def encode_image(self, image_file):
        if not os.path.isfile(image_file):
            raise FileNotFoundError(f"图片文件 {image_file} 不存在")
        
        img = Image.open(image_file).convert("L").resize(self.target_size)
        img_array = np.asarray(img)
        return self._hash_algo(img_array)

    def _hash_algo(self, image_array):
        dct_coef = dct(dct(image_array, axis=0), axis=1)
        
        dct_reduced_coef = dct_coef[: self.__coefficient_extract[0], : self.__coefficient_extract[1]]

        median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])

        hash_mat = dct_reduced_coef >= median_coef_val
        return self._array_to_hash(hash_mat)

    def find_duplicates(self, image_dir, max_distance_threshold=10):
        if not os.path.isdir(image_dir):
            raise NotADirectoryError(f"{image_dir} 不是一个有效的目录")

        encoding_map = {}
        for img_path in Path(image_dir).glob("*"):
            if img_path.is_file() and img_path.suffix.lower() != ".json":
                try:
                    hash_str = self.encode_image(str(img_path))
                    encoding_map[str(img_path)] = hash_str
                except Exception as e:
                    print(f"处理图像 {img_path} 出错: {str(e)}")

        duplicates = {}
        for img1, hash1 in encoding_map.items():
            for img2, hash2 in encoding_map.items():
                if img1 != img2 and self.hamming_distance(hash1, hash2) <= max_distance_threshold:
                    duplicates.setdefault(img1, []).append(img2)
        return duplicates
    
    @staticmethod
    def hamming_distance(hash1, hash2):
        hash1_bin = bin(int(hash1, 16))[2:].zfill(64) 
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return sum(i != j for i, j in zip(hash1_bin, hash2_bin))

def deduplicate_images(folder, similarity_threshold):
    print("开始图片去重...")
    phash = PHash()
    duplicates = phash.find_duplicates(folder, max_distance_threshold=similarity_threshold)
    
    for original, duplicate_list in duplicates.items():
        original_path = os.path.join(folder, os.path.basename(original))
        try:
            max_size = os.path.getsize(original_path)
            max_image = original
        except FileNotFoundError:
            print(f"文件 {original} 不存在,跳过...")
            continue
        
        for duplicate in duplicate_list:
            duplicate_path = os.path.join(folder, os.path.basename(duplicate))
            try:
                duplicate_size = os.path.getsize(duplicate_path)
                
                if duplicate_size > max_size:
                    max_size = duplicate_size
                    max_image = duplicate
            except FileNotFoundError:
                print(f"文件 {duplicate} 不存在,跳过...")
        
        for duplicate in duplicate_list:
            if duplicate != max_image:
                duplicate_path = os.path.join(folder, os.path.basename(duplicate))
                try:
                    os.remove(duplicate_path)
                    print(f"删除重复图片: {os.path.basename(duplicate)} (保留: {os.path.basename(max_image)})")
                except FileNotFoundError:
                    print(f"文件 {duplicate} 不存在,跳过...")
    print("图片去重完成")


def process_image_file(image_path, model_name_classify='mobilenetv3_v1.3_dist', model_name_real='mobilenetv3_v1.2_dist', aesthetic_model=_DEFAULT_MODEL, anime_style_age_model='mobilenetv3_v0_dist', monochrome_model='mobilenetv3_large_100_dist_safe2'):
    image = Image.open(image_path)
    base_size = 1024
    img_ratio = image.size[0] / image.size[1]  # width / height
    if img_ratio > 1:  # Width is greater than height
        new_size = (base_size, int(base_size / img_ratio))
    else:  # Height is greater than width or equal
        new_size = (int(base_size * img_ratio), base_size)
    image = image.resize(new_size, Image.Resampling.LANCZOS) 
    classify_scores = anime_classify_score(image, model_name=model_name_classify)
    real_scores = anime_real_score(image, model_name=model_name_real)
        # 添加anime_style_age_score功能
    anime_style_age_scores = anime_style_age_score(image, model_name=anime_style_age_model)
    # 添加get_monochrome_score功能
    monochrome_score = get_monochrome_score(image, model_name=monochrome_model)
    aesthetic_image = image.resize((image.size[0]//2, image.size[1]//2), Image.Resampling.LANCZOS)
    weighted_mean, percentile, scores_by_class =  _fn_predict(aesthetic_image, aesthetic_model)
    

    
    return image_path, {
        "imgscore": classify_scores,
        "anime_real_score": real_scores,
        "aesthetic_score": weighted_mean,
        "percentile": percentile,
        "scores_by_class": scores_by_class,
        "anime_style_age_score": anime_style_age_scores,
        "monochrome_score": monochrome_score
    }

def is_bad_image(image_scores, real_threshold, aesthetic_threshold, monochrome_threshold):
    imgscore = image_scores["imgscore"]
    anime_real_score = image_scores["anime_real_score"]
    aesthetic_score = image_scores["aesthetic_score"]
    scores_by_class = image_scores["scores_by_class"]
    anime_style_age_score = image_scores["anime_style_age_score"]
    monochrome_score = image_scores["monochrome_score"]
    bad_anime_style_ages = ["1970s-", "1980s", "1990s", "2000s"]
    bad_imgscore_types = ["not_painting", "3d"]
    
    return (
        max(imgscore, key=imgscore.get) in bad_imgscore_types or
        anime_real_score["real"] > real_threshold or
        aesthetic_score < aesthetic_threshold or
        
        max(anime_style_age_score, key=anime_style_age_score.get) in bad_anime_style_ages or
        monochrome_score > monochrome_threshold
    )

def main(args):
    # 第一步:收集图片
    with open(os.getcwd() + r"\web.txt", "r") as file:
        content = file.read()
        tasks = content.split("##SPILTED@")[1:] 

    gallery_dl_path = os.getcwd() + r"\gallery-dl\gallery-dl.exe"

    task_file = os.getcwd() + r"\task.txt"
    if os.path.exists(task_file):
        with open(task_file, "r") as file:
            task_lines = file.readlines()
            if task_lines and "Finished_a2498eagp3q!" not in task_lines:
                last_task = task_lines[-1].strip()
                if last_task in [task.split("\n")[0].strip() for task in tasks]: 
                    tasks = tasks[tasks.index(next(task for task in tasks if task.startswith(last_task))) + 1:]  
                else:
                    print("Error: Task not found in web.txt")
                    return

    for task in tasks: 
        lines = task.strip().split("\n")  
        output_dir_name = lines[0].strip()  
        urls = [url.strip() for url in lines[1:] if url.strip()] 
        print(f"Processing task: {output_dir_name}")

        for url in urls:
            print(url)
            command_gall = gallery_dl_path + f' "{url}" --write-metadata'
            os.system(command_gall)

        temp_folder = os.getcwd() + r"\temp"
        os.makedirs(temp_folder, exist_ok=True)

        for root, _, files in os.walk(os.getcwd() + r"\gallery-dl"):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    src_path = os.path.join(root, file)
                    json_path = src_path + ".json"
                    file_name, file_ext = os.path.splitext(file)
                    dst_path = os.path.join(temp_folder, file_name + '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + file_ext)
                    dst_json_path = dst_path + ".json"
                    shutil.move(src_path, dst_path)
                    if os.path.exists(json_path):
                        shutil.move(json_path, dst_json_path)

        extra_folder = os.getcwd() + r"\Extra"
        if os.path.exists(extra_folder):
            for root, _, files in os.walk(extra_folder):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        src_path = os.path.join(root, file)
                        json_path = src_path + ".json"
                        file_name, file_ext = os.path.splitext(file)
                        dst_path = os.path.join(temp_folder, file_name + '_' + ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + file_ext)
                        dst_json_path = dst_path + ".json"
                        shutil.move(src_path, dst_path)
                        if os.path.exists(json_path):
                            shutil.move(json_path, dst_json_path)

        # 第二步:标记及处理图片
        deduplicate_images(temp_folder, args.similarity_threshold)
        image_scores = {}
        print("Rating Image")
        with ProcessPoolExecutor(max_workers=2) as executor:
            results = executor.map(process_image_file, [os.path.join(temp_folder, image_name) for image_name in os.listdir(temp_folder) if not image_name.endswith(".json")])
            for image_path, scores in results:
                image_scores[image_path] = scores

        # 第三步:移动图片  
        output_folder = os.getcwd() + f"\\{output_dir_name}"
        bad_image_folder = os.path.join(output_folder, os.getcwd() + f"\\{output_dir_name}\\BadImage")
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(bad_image_folder, exist_ok=True)

        for image_path, scores in image_scores.items():
            json_path = image_path + ".json"
            if is_bad_image(scores, args.real_threshold, args.aesthetic_threshold, args.monochrome_threshold):
                shutil.move(image_path, os.path.join(bad_image_folder, os.path.basename(image_path)))
                if os.path.exists(json_path):
                    shutil.move(json_path, os.path.join(bad_image_folder, os.path.basename(json_path)))
            else:
                shutil.move(image_path, os.path.join(output_folder, os.path.basename(image_path)))
                if os.path.exists(json_path):
                    shutil.move(json_path, os.path.join(output_folder, os.path.basename(json_path)))

        with open(os.path.join(output_folder, "crawler.txt"), "w") as file:
            file.write(task)

        with open(task_file, "a") as file:
            file.write(output_dir_name + "\n")

        # 清空temp文件夹
        for file_name in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    with open(task_file, "a") as file:
        file.write("Finished_a2498eagp3q!")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image processing script')
    parser.add_argument('--real_threshold', type=float, default=0.9, help='Real threshold for anime_real_score')
    parser.add_argument('--aesthetic_threshold', type=float, default=0.35, help='Aesthetic threshold for aesthetic_score')
    parser.add_argument('--monochrome_threshold', type=float, default=0.9, help='Monochrome threshold for monochrome_score')
    parser.add_argument('--similarity_threshold', type=float, default=0.2, help='Similarity threshold for image deduplication')
    args = parser.parse_args()

    main(args)
    print("Finished!")

    gallery_dl_folder = os.getcwd() + r"\gallery-dl"
    if os.path.exists(gallery_dl_folder):
        for root, dirs, _ in os.walk(gallery_dl_folder, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted folder: {dir_path}")
                except OSError as e:
                    print(f"Error deleting folder: {dir_path}" + f"Error message: {e}")

    task_file = os.path.join(os.getcwd(), "task.txt")
    if os.path.exists(task_file):
        try:
            os.remove(task_file)
            print(f"Deleted file: {task_file}")
        except OSError as e:
            print(f"Error deleting file: {task_file}" + f"Error message: {e}")