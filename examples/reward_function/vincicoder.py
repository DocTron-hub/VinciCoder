# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any
import os
import uuid
import subprocess
import tempfile
import sys
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel
import torch.nn.functional as F
from typing import List, Dict
import math
import cairosvg
from concurrent.futures import TimeoutError
from tqdm import tqdm
import multiprocessing
from playwright.sync_api import sync_playwright
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn as nn

model = AutoModelForImageClassification.from_pretrained('facebook/dinov2-large').cuda()
NUM_RENDER_PROCESS = 64
playwright_context = None
script_path = Path(__file__).resolve()
current_dir = script_path.parent
parent_parent_dir = current_dir.parent.parent

temp_images_path = parent_parent_dir / 'temp_images'
temp_images_path.mkdir(exist_ok=True, exist_ok=True)

if torch.cuda.device_count() >= 2:
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
model.eval()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def extract_code_names(prompt: str) -> list[str]:
    """
    Helper function: checks if the given prompt text contains specific code name keywords.
    If no keywords are found, it defaults to returning ["python"].
    """
    code_names_to_check = ["svg", "html", "python", "latex", "mermaid", "tikz"]
    lower_prompt = prompt.lower()
    found_names = [name for name in code_names_to_check if name in lower_prompt]
    
    # If found_names is not empty, return it; otherwise, return ["python"]
    return found_names if found_names else ["python"]

def extract_code(language_name, response):
    if not language_name:
        return (None, [])
    pattern = re.compile(rf"```{language_name}\s*(.*?)\s*```", re.DOTALL)
    matches = pattern.findall(response)
    
    return (language_name, matches)

def cal_format_reward(found_code_list: str) -> float:
    if found_code_list:
        return 1.0
    else:
        return 0.0

def render_python_code(code: str, save_path: str):
    # Create a temporary Python file
    file_path = os.path.join(TEMP_DIR, f"{Path(save_path).stem}.py")
    
    # Modify the code to save the image to the specified path
    modified_code = code
    if "rdkit" in code:
        # Define the pattern and replacement for RDKit's drawing function
        pattern = r'(drawer\.WriteDrawingText\s*\()(.*?)(\))'
        replacement = fr'\1"{save_path}"\3'
        
        # Apply the regex substitution
        modified_code = re.sub(pattern, replacement, code)
    elif "indigo" in code:
        pattern = r'(renderer\.renderToFile\s*\(\s*[^,]+,\s*)(.*?)(\))'
        replacement = fr'\1"{save_path}"\3'
        modified_code = re.sub(pattern, replacement, code)
    elif "fig.write_image" in code:
        pattern = r'(fig\.write_image\s*\()([\'"].*?[\'"]|[a-zA-Z0-9_]+)'
        modified_code = re.sub(pattern, f'\\1r"{save_path}"', code)
    elif "fig.savefig" in code:
        pattern = r'(fig\.savefig\s*\()([\'"].*?[\'"]|[a-zA-Z0-9_]+)'
        modified_code = re.sub(pattern, f'\\1r"{save_path}"', code)
    elif "fig.show()" in code:
        modified_code = code.replace("fig.show()", f'fig.savefig(r"{save_path}")\n')
    elif "plt.savefig" in code:
        pattern = r'(plt\.savefig\s*\()([\'"].*?[\'"]|[a-zA-Z0-9_]+)'
        modified_code = re.sub(pattern, f'\\1r"{save_path}"', code)
    elif "plt.show()" in code:
        modified_code = code.replace("plt.show()", f'plt.savefig(r"{save_path}")\n')
    else:
        # If no save/show command is found, append one. Assumes `plt` is used.
        modified_code += f'\nimport matplotlib.pyplot as plt\nplt.savefig(r"{save_path}")'

    # Save the modified code to the temporary file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_code)
    
    # Execute the code to render the image
    try:
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=10,
            check=False  # Do not raise exception on non-zero exit codes
        )
        
        success = result.returncode == 0 and os.path.exists(save_path)
        if success:
            return True, "Success: Image rendered."
        else:
            error_msg = result.stderr if result.stderr else "An unknown error occurred during script execution."
            return False, f"Failed to render image. Error: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, "Failed to render image: The script took too long to execute."
    except Exception as e:
        return False, f"Failed to render image with an exception: {str(e)}"
    finally:
        # Clean up the temporary Python file
        if os.path.exists(file_path):
            os.remove(file_path)

def render_latex_code(code: str, save_path: str):
    if not save_path.lower().endswith('.png'):
        save_path += '.png'

    # Define temporary file paths based on the save_path stem, inside the system's temp directory.
    TEMP_DIR = tempfile.gettempdir()
    base_name = Path(save_path).stem
    tex_path = os.path.join(TEMP_DIR, f"{base_name}.tex")
    pdf_path = os.path.join(TEMP_DIR, f"{base_name}.pdf")
    log_path = os.path.join(TEMP_DIR, f"{base_name}.log")
    aux_path = os.path.join(TEMP_DIR, f"{base_name}.aux") # Also track .aux for cleanup

    # Intelligently handle the input code
    if "\\documentclass" in code:
        # If it's a full document, check if it's an 'article' that needs replacement
        pattern = r"\\documentclass(\[.*?\])?\s*\{article\}"
        replacement = r"\\documentclass[tikz, border=3.14mm]{standalone}"
        
        modified_code, num_replacements = re.subn(pattern, replacement, code, count=1)
        
        if num_replacements > 0:
            full_code = modified_code
        else:
            full_code = code
    else:
        # If it's a code snippet, wrap it in our default template
        full_code = f"""
        \\documentclass[preview, border=3.14mm]{{standalone}}
        \\usepackage{{amsmath}}
        \\usepackage{{amssymb}}
        \\usepackage{{graphicx}}
        \\usepackage{{xcolor}}
        \\usepackage{{tikz}}
        \\usepackage{{pgfplots}}
        \\pgfplotsset{{compat=1.17}}
        \\begin{{document}}
        {code}
        \\end{{document}}
        """

    # Save the prepared LaTeX code to the temporary .tex file
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(full_code)

    try:
        # Execute pdflatex to compile the .tex file into a .pdf
        result = subprocess.run(
            ['/home/hadoop-basecv/texlive/2025/bin/x86_64-linux/pdflatex', '-interaction=nonstopmode', '-output-directory', TEMP_DIR, tex_path],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            error_info = f"pdflatex failed with return code {result.returncode}.\n"
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as log_file:
                    error_info += "LaTeX Log Tail:\n" + "".join(log_file.readlines()[-20:])
            else:
                error_info += "Stderr:\n" + result.stderr
            return False, f"Render failed during LaTeX compilation. Error: {error_info}"

        if not os.path.exists(pdf_path):
            return False, "Render failed: PDF file was not created by pdflatex."

        # Convert the generated PDF to a PNG image
        images = convert_from_path(pdf_path, dpi=300)

        if not images:
            return False, "Render failed: pdf2image could not convert the PDF."
        
        images[0].save(save_path, 'PNG')

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True, "Success: Image rendered."
        else:
            return False, "Render failed: Output PNG file is empty or was not created."

    except FileNotFoundError:
        return False, "Render failed: 'pdflatex' command not found. Is TeX Live installed and in your PATH?"
    except subprocess.TimeoutExpired:
        return False, "Render failed: LaTeX compilation took too long to execute."
    except Exception as e:
        return False, f"Render failed with an unexpected exception: {str(e)}"
    finally:
        # Clean up all temporary intermediate files
        for path in [tex_path, pdf_path, log_path, aux_path]:
            if os.path.exists(path):
                os.remove(path)

def render_svg_code(code: str, save_path: str):
    if not save_path.lower().endswith('.png'):
        save_path += '.png'
    try:
        cairosvg.svg2png(
            bytestring=code.encode('utf-8'),
            write_to=save_path,
            output_width=336,
            output_height=336,
            background_color="white"
        )
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True, "Success"
        else:
            return False, "Render failed: Empty Output File"
    except Exception as e:
        return False, f"Render failed: {e}"


def init_worker_playwright():
    global playwright_context
    try:
        playwright_context = sync_playwright().start()
        # Optional: You can print this to see it running once per worker
        # print(f"[Process {os.getpid()}] Playwright initialized.")
    except Exception as e:
        print(f"Failed to initialize Playwright in process {os.getpid()}: {e}")

def render_html_code(code: str, save_path: str):
    """
    Optimized version that uses a pre-initialized Playwright context 
    from a global variable within the worker process.
    """
    # Safety check in case the initializer failed
    if playwright_context is None:
        return False, f"Render failed: Playwright not initialized in process {os.getpid()}."

    if not save_path.lower().endswith('.png'):
        save_path += '.png'

    try:
        # NO "with sync_playwright() as p:" anymore.
        # We launch a browser from the existing context.
        browser = playwright_context.chromium.launch()
        page = browser.new_page()

        # Set the page content directly from the HTML string
        page.set_content(code, wait_until="load", timeout=6000)

        # Take a screenshot of the full rendered page
        page.screenshot(path=save_path, full_page=True, animations="disabled", timeout=6000)
        
        browser.close()

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return True, "Success: Image rendered successfully."
        else:
            return False, "Render failed: Output file is empty or was not created."
            
    except Exception as e:
        # It's good practice to return the error rather than printing from the worker
        # but keeping your original logic for now.
        error_details = f"Render failed in process {os.getpid()}: {type(e).__name__}: {e}"
        # traceback.print_exc() 
        return False, error_details

def encode_image_to_base64(image_path):
    if not os.path.exists(image_path):
        return None
    
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def is_valid_image(filepath, max_pixels=83000000):
    if not filepath or not os.path.isfile(filepath):
        return False
    
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            if width * height > max_pixels:
                print(f"Error: Image size ({width}x{height}) exceeds the maximum allowed pixels ({max_pixels}).")
                return False
            img.verify()
            
        return True
    except (IOError, SyntaxError, UnidentifiedImageError) as e:
        print(f"Error: Invalid or corrupt image file at '{filepath}'. Details: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def cal_sim_batched(original_images: List[Image.Image], generated_img_paths: List[str]) -> List[float]:
    """
    Calculates cosine similarity between two BATCHES of images where each item
    may contain a variable number of frames (variable first dimension).

    It concatenates all frames into a single batch, computes similarity,
    and then averages the scores for each original item.
    """
    with torch.inference_mode():
        original_tensors = []
        generated_tensors = []
        # We need to store the length of the first dimension of each tensor
        original_lengths = []

        if not original_images:
            return []

        for original_img, gen_path in zip(original_images, generated_img_paths):
            target_size = original_img.size
            generated_img = Image.open(gen_path).convert('RGB')
            resized_gen_img = generated_img.resize(target_size, Image.Resampling.BICUBIC)
            
            # load_image now returns a tensor like [X, 3, H, W]
            original_tensor = load_image(original_img)
            generated_tensor = load_image(resized_gen_img)
            
            # Store the tensors
            original_tensors.append(original_tensor)
            generated_tensors.append(generated_tensor)
            
            # CRITICAL: Record the length of the first dimension
            original_lengths.append(original_tensor.shape[0])
            
            # Optional: Add a check to ensure generated tensor has a matching number of frames
            if original_tensor.shape[0] != generated_tensor.shape[0]:
                print(f"Warning: Mismatch in number of frames for an item. "
                      f"Original has {original_tensor.shape[0]}, "
                      f"Generated has {generated_tensor.shape[0]}.")


        # Instead of torch.stack, we use torch.cat along the first dimension (dim=0)
        original_batch = torch.cat(original_tensors, dim=0).to(torch.bfloat16).cuda()
        generated_batch = torch.cat(generated_tensors, dim=0).to(torch.bfloat16).cuda()

        # InternViT 
        # with torch.no_grad():
        #     original_hidden_state = model(original_batch).last_hidden_state
        #     generated_hidden_state = model(generated_batch).last_hidden_state

        # DINOv2 
        with torch.no_grad():
            original_hidden_state = model(pixel_values=original_batch, output_hidden_states=True).hidden_states[-1]
            generated_hidden_state = model(pixel_values=generated_batch, output_hidden_states=True).hidden_states[-1]
        
        original_cls_features = original_hidden_state[:, 0, :]
        generated_cls_features = generated_hidden_state[:, 0, :]

        sim = F.cosine_similarity(
            original_cls_features,
            generated_cls_features,
            dim=1
        )

        normalized_sim = (sim + 1) / 2
        score_chunks = torch.split(normalized_sim, original_lengths)
        average_scores = [chunk.mean().item() for chunk in score_chunks]
        
        return average_scores

def cal_visual_reward_batched(visual_inputs: List[Dict], batch_size: int = 64) -> List[float]:
    num_inputs = len(visual_inputs)

    tasks_to_run = []
    for i, item in enumerate(visual_inputs):
        lang = item['language_name']
        response_code = item['code']
        
        if lang in ["python", "svg", "html", "latex"]:
            unique_id = str(uuid.uuid4())
            # Using a language-specific extension for clarity, though the original used .png
            generated_img_path = os.path.join(TEMP_DIR, f"generated_{lang}_{unique_id}.png")
            
            tasks_to_run.append({
                'language': lang,
                'code': response_code,
                'path': generated_img_path,
                'original_image': item['image'],
                'original_index': i
            })

    all_generated_paths = [task['path'] for task in tasks_to_run]

    try:
        print(f"-> Starting rendering for {len(tasks_to_run)} items (Python/SVG/HTML/LaTeX)...")
        
        python_success, python_failure = 0, 0
        svg_success, svg_failure = 0, 0
        html_success, html_failure = 0, 0
        latex_success, latex_failure = 0, 0
        
        # Use multiprocessing.Pool as a context manager
        with multiprocessing.Pool(processes=NUM_RENDER_PROCESS, initializer=init_worker_playwright) as pool:
            async_results = []
            for task in tasks_to_run:
                lang = task['language']
                res = None
                if lang == 'python':
                    res = pool.apply_async(render_python_code, args=(task['code'], task['path']))
                elif lang == 'svg':
                    res = pool.apply_async(render_svg_code, args=(task['code'], task['path']))
                elif lang == 'html':
                    res = pool.apply_async(render_html_code, args=(task['code'], task['path']))
                elif lang == 'latex':
                    res = pool.apply_async(render_latex_code, args=(task['code'], task['path']))
                
                if res:
                    async_results.append({'result_obj': res, 'task': task})
            
            # --- Get and process results ---
            temp_results = {}
            RENDER_TIMEOUT = 5 # seconds

            print(f"-> All {len(async_results)} tasks submitted. Waiting for results...")
            for item in tqdm(async_results, desc="🎨 Processing Tasks"):
                async_result = item['result_obj']
                task = item['task']
                lang = task['language']
                
                try:
                    # Use .get(timeout=...) to wait for the result
                    result_data = async_result.get(timeout=RENDER_TIMEOUT)
                    temp_results[task['path']] = result_data
                    
                    if result_data[0]:  # Success case
                        if lang == 'python': python_success += 1
                        elif lang == 'svg': svg_success += 1
                        elif lang == 'html': html_success += 1
                        elif lang == 'latex': latex_success += 1
                    else:  # Failure case
                        if lang == 'python': python_failure += 1
                        elif lang == 'svg': svg_failure += 1
                        elif lang == 'html': html_failure += 1
                        elif lang == 'latex': latex_failure += 1
                
                except TimeoutError:
                    tqdm.write(f"\n[ERROR] Task '{task['path']}' timed out after {RENDER_TIMEOUT} seconds and was cancelled.")
                    temp_results[task['path']] = (False, "Task timed out.")
                    if lang == 'python': python_failure += 1
                    elif lang == 'svg': svg_failure += 1
                    elif lang == 'html': html_failure += 1
                    elif lang == 'latex': latex_failure += 1

                except Exception as exc:
                    tqdm.write(f"\n[ERROR] Task '{task['path']}' failed with an exception: {exc}")
                    temp_results[task['path']] = (False, str(exc))
                    if lang == 'python': python_failure += 1
                    elif lang == 'svg': svg_failure += 1
                    elif lang == 'html': html_failure += 1
                    elif lang == 'latex': latex_failure += 1

        results = []
        for task in tasks_to_run:
            results.append(temp_results.get(task['path'], (False, "Task was not processed.")))

        print("\n-> Rendering finished.")
        print("\n--- Summary ---")
        print(f"Python Success: {python_success}, Python Failure: {python_failure}")
        print(f"SVG Success:    {svg_success}, SVG Failure:    {svg_failure}")
        print(f"HTML Success:   {html_success}, HTML Failure:   {html_failure}")
        print(f"LaTeX Success:  {latex_success}, LaTeX Failure:  {latex_failure}")
        print("---------------")

        successful_original_images = []
        successful_generated_paths = []
        indices_of_success = []
        for task, (gen_success, _) in zip(tasks_to_run, results):
            if gen_success and is_valid_image(task['path']):
                successful_original_images.append(task['original_image'])
                successful_generated_paths.append(task['path'])
                indices_of_success.append(task['original_index'])

        num_successful = len(indices_of_success)
        calculated_rewards = []
        if num_successful > 0:
            total_batches = math.ceil(num_successful / batch_size)
            for i in range(0, num_successful, batch_size):
                print(f"  -> Processing similarity mini-batch {i//batch_size + 1}/{total_batches}...")
                
                start_index = i
                end_index = min(i + batch_size, num_successful)
                
                original_chunk = successful_original_images[start_index:end_index]
                generated_chunk = successful_generated_paths[start_index:end_index]
                
                rewards_chunk = cal_sim_batched(original_chunk, generated_chunk)
                calculated_rewards.extend(rewards_chunk)
                current_average = np.mean(calculated_rewards)
                ### Check Use
                current_variance = np.var(calculated_rewards)
                print(f"     Length of reward==1: {calculated_rewards.count(1)}")
                print(f"     Current average reward: {current_average:.10f}")
                print(f"     Current variance of reward: {current_variance:.10f}, min value {np.min(calculated_rewards)}, max value {np.max(calculated_rewards)}")
    finally:
        for path in all_generated_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"Error removing file {path}: {e}")
                
    final_rewards = [0.0] * num_inputs
    for i, reward in zip(indices_of_success, calculated_rewards):
        final_rewards[i] = reward
        
    return final_rewards

def compute_score(reward_inputs, format_weight=0.1):
    if not isinstance(reward_inputs, list):
        raise ValueError("Input must be a list of dictionaries.")

    num_inputs = len(reward_inputs)
    format_scores = [0.0] * num_inputs
    visual_inputs_to_process = []
    # This list stores the original index of the items that need visual scoring
    indices_needing_visual_score = []

    # This loop runs quickly, preparing data for the slow batch operation.
    print("--- Stage 1: Pre-processing on CPU ---")
    for i, reward_input in enumerate(reward_inputs):
        language_name = extract_code_names(reward_input["problem"])[0]
        if language_name == 'tikz':
            language_name = 'latex'
        language_name, found_code_list = extract_code(language_name, reward_input["response"])
        
        # Calculate format score for every item
        current_format_score = cal_format_reward(found_code_list)
        format_scores[i] = current_format_score
        
        # Only gather data for visual reward if the format score is not zero
        if current_format_score > 0.0 and found_code_list:
            visual_inputs_to_process.append({
                "language_name": language_name,
                "code": found_code_list[-1],
                "image": reward_input["multi_modal_data"]["images"][0]
            })
            indices_needing_visual_score.append(i)

    print(f"Found {len(visual_inputs_to_process)} items needing visual scoring.")

    print("--- Stage 2: Calculating on GPU ---")
    batched_visual_scores = []
    if visual_inputs_to_process:
        # This function takes a list and processes them all at once on the GPU
        batched_visual_scores = cal_visual_reward_batched(visual_inputs_to_process)

    final_scores = []
    # Create a full list of visual scores, initialized to 0.0
    visual_scores = [0.0] * num_inputs
    
    # Place the calculated visual scores back into their original positions
    for original_index, score in zip(indices_needing_visual_score, batched_visual_scores):
        visual_scores[original_index] = score

    print("--- Stage 3: Format Rewards ---")
    # Now, calculate the final combined score for all items
    for i in range(num_inputs):
        fs = format_scores[i]
        vs = visual_scores[i]
        overall_score = (1 - format_weight) * vs + format_weight * fs
        execution_score = 1.0 if vs != 0 else 0.0
        
        final_scores.append({
            "overall": overall_score,
            "format": fs,
            "accuracy": vs, 
            "execution":  execution_score
        })

    return final_scores