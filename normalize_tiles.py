import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from multiprocessing import Pool, cpu_count

# Function to load all images from a directory and its subdirectories
def load_images_from_folder(folder):
    images = []
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img)
                images.append((os.path.relpath(img_path, folder), img_np))  # store relative path
    return images

# Function to save images to a directory, preserving subdirectory structure
def save_images_to_folder(images, folder):
    for relative_path, img_np in images:
        output_path = os.path.join(folder, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        img.save(output_path)

# Function to extract amplitude
def extract_amp(img_np):
    fft = np.fft.fft2(img_np, axes=(-2, -1))
    amp_np = np.abs(fft)
    return amp_np

# Function to mutate amplitudes
def mutate(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))
    h = a_src.shape[1]
    w = a_src.shape[2]

    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_src[:, h1:h2, w1:w2] = a_trg[:, h1:h2, w1:w2]
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src

# Function to normalize images using the global average amplitude
def normalize_image(args):
    img_np, amp_trg, L = args
    fft_src_np = np.fft.fft2(img_np, axes=(-2, -1))
    amp_src = np.abs(fft_src_np)
    pha_src = np.angle(fft_src_np)
    amp_src_ = mutate(amp_src, amp_trg, L=L)
    fft_src_ = amp_src_ * np.exp(1j * pha_src)
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg_real = np.real(src_in_trg)
    return src_in_trg_real

# Function to compute the average amplitude for a batch of images
def compute_batch_avg_amp(batch):
    amp_list = [extract_amp(img_np) for _, img_np in batch]
    return np.mean(amp_list, axis=0)

# Process images in a single folder to compute average amplitude
def process_folder_for_avg_amp(input_folder, batch_size):
    images = load_images_from_folder(input_folder)
    if not images:
        print(f"No images to process in {input_folder}. Exiting.")
        return None

    avg_amps = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        avg_amps.append(compute_batch_avg_amp(batch))
    
    folder_avg_amp = np.mean(avg_amps, axis=0)
    return folder_avg_amp

# Process images in a single folder for normalization
def process_folder_for_normalization(input_folder, output_folder, global_avg_amp, batch_size, num_workers):
    images = load_images_from_folder(input_folder)
    if not images:
        print(f"No images to process in {input_folder}. Exiting.")
        return

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        normalize_args = [(img_np, global_avg_amp, 0.1) for _, img_np in batch]
        with Pool(num_workers) as pool:
            normalized_images = pool.map(normalize_image, normalize_args)

        normalized_images_with_paths = [(img_path, img_np) for (img_path, _), img_np in zip(batch, normalized_images)]
        save_images_to_folder(normalized_images_with_paths, output_folder)

# Main processing function
def process_tiles(input_folders, output_folders, batch_size=100, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()

    # Step 1: Compute average amplitude for each folder
    folder_avg_amps = []
    for input_folder in input_folders:
        subfolders = [os.path.join(input_folder, subfolder) for subfolder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, subfolder))]
        for subfolder in subfolders:
            print(f'Processing {subfolder}.')
            folder_avg_amp = process_folder_for_avg_amp(subfolder, batch_size)
            if folder_avg_amp is not None:
                folder_avg_amps.append(folder_avg_amp)

    # Compute global average amplitude
    if folder_avg_amps:
        global_avg_amp = np.mean(folder_avg_amps, axis=0)
        print("Global average amplitude calculated.")
    else:
        print("No average amplitudes were computed. Exiting.")
        return

    # Step 2: Normalize images using the global average amplitude
    for input_folder, output_folder in zip(input_folders, output_folders):
        subfolders = [os.path.join(input_folder, subfolder) for subfolder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, subfolder))]
        output_subfolders = [os.path.join(output_folder, os.path.basename(subfolder)) for subfolder in subfolders]

        for input_subfolder, output_subfolder in zip(subfolders, output_subfolders):
            process_folder_for_normalization(input_subfolder, output_subfolder, global_avg_amp, batch_size, num_workers)

# Example usage
input_folders = ['/work/u6658716/TCGA-LUAD/dsmil-wsi/WSI/TCGA-lung_ours/single/0','/work/u6658716/TCGA-LUAD/dsmil-wsi/WSI/TCGA-lung_ours/single/1','/work/u6658716/TCGA-LUAD/dsmil-wsi/WSI/TCGA-lung_test/single/0','/work/u6658716/TCGA-LUAD/dsmil-wsi/WSI/TCGA-lung_test/single/1']
output_folders = ['/work/u6658716/TCGA-LUAD/dsmil-wsi/WSI/TCGA-lung_ours/single/norm_0','/work/u6658716/TCGA-LUAD/dsmil-wsi/WSI/TCGA-lung_ours/single/norm_1','/work/u6658716/TCGA-LUAD/dsmil-wsi/WSI/TCGA-lung_test/single/norm_0','/work/u6658716/TCGA-LUAD/dsmil-wsi/WSI/TCGA-lung_test/single/norm_1']
process_tiles(input_folders, output_folders)