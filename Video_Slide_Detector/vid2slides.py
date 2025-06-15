import argparse
import collections
import cv2
from decord import VideoReader
from decord import cpu
import glob
import ffmpeg
import json
from matplotlib import image
import numpy as np
import os
import pickle
import pytesseract
from pytesseract import Output
import subprocess
import sklearn
import sklearn.cluster
import tempfile
from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update for Windows users

def get_video_info(path):
    probe = ffmpeg.probe(path)

    for stream in probe["streams"]:
        if stream["codec_type"] == "video":
            return stream


def to_timestamp(ts):
    h, m, s = ts // 3600, (ts // 60) % 60, int(ts % 60)
    return f'{h:02}:{m:02}:{s:02}'


def log_viterbi(log_B, log_A, log_pi):
    """Computes the Viterbi path.

    Note that this function uses numerically stable logs of probability.

    Arguments:
    log_B: A matrix of size T X N, where N is the number of states.
        It's the log probability of the observation at time T given that the
        system was in state N.
    log_A: The state-transition probability matrix, size NxN
    log_pi: The initial state distribution vector pi = {pi_1....pi_N}
    """
    delta = log_pi + log_B[0, :]
    phi = np.zeros(log_B.shape, dtype=np.uint16)

    for t in range(1, log_B.shape[0]):
        projected = delta.reshape((-1, 1)) + log_A
        delta = np.max(projected, axis=0) + log_B[t, :]
        phi[t, :] = np.argmax(projected, axis=0)

    q = np.zeros(log_B.shape[0], dtype=int)
    q[-1] = np.argmax(delta)
    for t in range(log_B.shape[0] - 2, -1, -1):
        q[t] = phi[t + 1, q[t + 1]]

    return q


def heuristic_frames(sizes, ban_time=5):
    """
    Get some heuristically chosen reference frames. 
    
    Pick the images which are least compressible, and ban surrounding images.
    """
    tuples = [(sz, x) for x, sz in enumerate(sizes)]
    banned = {}

    chosen = []
    for _, x in sorted(tuples)[::-1]:
        if x in banned:
            continue

        for d in range(-ban_time, ban_time + 1):
            banned[x + d] = 1

        chosen.append(x)

    return chosen


def extract_thumbnails(video, 
                       lo_dir,                        
                       lo_size=(320, 180), 
                       thumb_interval=2):
    """
    Extract thumbnails from video and output them to output dir.

    Arguments:
        video: the video path
        lo_dir: the output directory for low-res thumbnails. Thumbnails are put 
            in this directory named thumb-%02d.jpg
        lo_size: the max size of the low-res thumbnails. We will resize the thumbnails
            to fit within this bounding box, preserving aspect ratio.
        thumb_interval (optional): the time in seconds between thumbnails
    """
    info = get_video_info(video)
    w, h = info['coded_width'], info['coded_height']

    # Dynamically calculate output size to preserve aspect ratio
    aspect_ratio = w / h
    max_w, max_h = lo_size
    if w / max_w > h / max_h:
        wo = max_w
        ho = int(max_w / aspect_ratio)
    else:
        ho = max_h
        wo = int(max_h * aspect_ratio)

    total_frames = int(float(info['duration']) / thumb_interval)

    # Ensure output directory exists
    if not os.path.exists(lo_dir):
        os.makedirs(lo_dir)

    with tqdm(total=total_frames, desc="Extracting thumbnails") as pbar:
        result = subprocess.run([
            'ffmpeg',
            '-y',
            '-i',
            video,
            '-vf',
            f'scale={wo}:{ho}:force_original_aspect_ratio=decrease',
            '-r',
            f'{1/thumb_interval}',
            '-f',
            'image2',
            os.path.join(lo_dir, 'thumb-%04d.jpg')
        ], shell=False, capture_output=True, text=True)
        pbar.update(total_frames)
        if result.returncode != 0:
            print(f"FFmpeg failed with error code {result.returncode}.")
            print("FFmpeg stderr:")
            print(result.stderr)
        else:
            print("FFmpeg completed successfully.")
        # Debug: List files in lo_dir
        import glob
        print(f"Checking for thumbnails in: {lo_dir}")
        print('Found files:', glob.glob(os.path.join(lo_dir, 'thumb-*.jpg')))


def extract_frames(video, hi_dir, hi_size, times):
    info = get_video_info(video)
    w, h = info['coded_width'], info['coded_height']

    # Dynamically calculate output size to preserve aspect ratio
    aspect_ratio = w / h
    max_w, max_h = hi_size
    if w / max_w > h / max_h:
        wo = max_w
        ho = int(max_w / aspect_ratio)
    else:
        ho = max_h
        wo = int(max_h * aspect_ratio)

    framerate = int(info['nb_frames']) / float(info['duration'])
    nframes = [int(framerate * (2 * (time + 1))) for time in times]

    vr = VideoReader(video, ctx=cpu(0))
    total_frames = len(vr)
    print(f"[extract_frames] Total frames in video: {total_frames}")
    print(f"[extract_frames] Requested frame indices: {nframes}")
    # Remove duplicates and sort
    frame_time_pairs = list(zip(nframes, times))
    # Filter out-of-bounds
    frame_time_pairs = [(f, t) for f, t in frame_time_pairs if 0 <= f < total_frames]
    # Remove duplicates by frame index, keeping the first occurrence
    seen = set()
    unique_pairs = []
    for f, t in frame_time_pairs:
        if f not in seen:
            seen.add(f)
            unique_pairs.append((f, t))
    if not unique_pairs:
        print("No valid frame indices to extract.")
        return
    valid_nframes, valid_times = zip(*unique_pairs)
    print(f"[extract_frames] Valid frame indices: {valid_nframes}")
    # Decord get_batch cannot handle large batches, so split into chunks
    chunk_size = 100
    for chunk_start in range(0, len(valid_nframes), chunk_size):
        chunk_nframes = list(valid_nframes)[chunk_start:chunk_start+chunk_size]
        chunk_times = list(valid_times)[chunk_start:chunk_start+chunk_size]
        print(f"[extract_frames] Processing chunk: {chunk_nframes}")
        try:
            frames = vr.get_batch(chunk_nframes).asnumpy()
        except Exception as e:
            print(f"[extract_frames] Error extracting frames for chunk {chunk_nframes}: {e}")
            continue
        with tqdm(total=len(chunk_nframes), desc="Extracting high-resolution frames") as pbar:
            for i, (idx, t) in enumerate(zip(chunk_nframes, chunk_times)):
                frame = frames[i, :, :, :]
                # Now clear why r and b are mixed up.
                frame = frame[:, :, np.array([2, 1, 0])]
                assert frame.ndim == 3
                assert frame.shape[-1] == 3

                cv2.imwrite(
                    os.path.join(hi_dir, f'thumb-{t+1:04}.png'),
                    cv2.resize(frame, (wo, ho))
                )
                pbar.update(1)


def get_delta_images(the_dir, has_face):
    matching_images = glob.glob(os.path.join(the_dir, 'thumb-*.jpg'))
    nimages = len(matching_images)
    if nimages == 0:
        raise ValueError(f"No thumbnails found in directory: {the_dir}. Check if thumbnail extraction succeeded.")
    sizes = []

    for i, filename in enumerate(matching_images):
        if i == 0:
            img = image.imread(filename)
            images = np.zeros((nimages, img.shape[0], img.shape[1]))

        sizes.append(os.stat(filename).st_size)
        img = image.imread(filename)
        images[i, :, :] = img.mean(axis=2)

        if has_face[i]:
            # Remove faces out of the pool of potential matches.
            sizes[i] = 0

    candidates = sorted(heuristic_frames(sizes))
    if not candidates:
        raise ValueError("No candidate frames found for slide detection. This may be due to all frames being filtered out or too few valid thumbnails.")

    candidate_images = np.stack([images[i, :, :] for i in candidates])
    assert candidate_images.shape[0] == len(candidates)
    assert candidate_images.ndim == 3

    delta_images = np.zeros((nimages, len(candidates)))
    for i in range(len(candidates)):
        delta_images[:, i] = ((images.reshape((images.shape[0], -1)) - 
            candidate_images[i, :, :].reshape((1, -1))) ** 2).sum(axis=1)
    
    return sizes, candidates, delta_images


def max_likelihood_sequence(nll, jump_probability=0.2):
    """
    Calculate the maximum likelihood sequence of images.

    Takes a sequence of images and attributes these images to a sequence of 
    template images. Uses a left-to-right HMM to attribute find the ML sequence
    of images.

    Arguments:
        nll: an (ntotal, ntemplate) np.array, where the (i, j)'th element 
        contains the negative log-likelihood of the j'th image as an instance of
        the i'th template. Under a Gaussian noise generative model, this would 
        be the sum-of-squares between template and instance.
        jump_probability: the probability that the sequence jumps from one 
        template to a further in the sequence.

    Returns:
        The maximum likelihood sequence of templates.
    """
    assert nll.shape[0] >= nll.shape[1]
    ncandidates = nll.shape[1]

    log_B = -nll / nll.mean()

    T = (np.arange(ncandidates).reshape((-1, 1)) < 
         np.arange(ncandidates).reshape((1, -1)))
    A = (1 - jump_probability) * np.eye(ncandidates) + jump_probability * T / T.sum(axis=1, keepdims=True)
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    A = A / A.sum(axis=1, keepdims=True)
    
    log_pi = np.log(np.ones(ncandidates) / ncandidates)
    
    seq = log_viterbi(log_B, np.log(A), log_pi)
    return seq


def extract_crop(info):
    """
    Find a reasonable crop given the information available.
    Downscale images and sample if too many to avoid memory errors.
    """
    import math
    ims = []
    slide_elements = [el for el in info['sequence'] if el['type'] == 'slide']
    max_samples = 50
    # Sample up to max_samples slides evenly spaced
    if len(slide_elements) > max_samples:
        indices = [math.floor(i * len(slide_elements) / max_samples) for i in range(max_samples)]
        slide_elements = [slide_elements[i] for i in indices]
    for el in slide_elements:
        im = cv2.imread(el['source'], cv2.IMREAD_GRAYSCALE)
        if im is not None:
            # Downscale to 1/4 size for memory efficiency
            im = cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4), interpolation=cv2.INTER_AREA)
            ims.append(im)
    if not ims:
        print("No slides found in the sequence. Returning default crop.")
        return (0, 0, 0, 0)
    ims = [im.astype(np.float32) / 255.0 for im in ims]
    A = np.stack(ims, axis=0)
    broad_crop = (A.mean(axis=0) > 0.2).astype(np.uint8)
    contours, _ = cv2.findContours(broad_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_ar = 0
    x = y = w = h = 0
    for contour in contours:
        ar = cv2.contourArea(contour)
        if ar > biggest_ar:
            biggest_ar = ar
            (x, y, w, h) = cv2.boundingRect(contour)
    # Scale crop back to original size
    scale = 4
    return (x * scale, y * scale, w * scale, h * scale)


def deduplicate_slides(slides, similarity_threshold=0.9):
    """
    Deduplicate slides based on visual similarity and OCR text content.

    Arguments:
        slides: List of slide dictionaries containing 'source' and 'text_ocr'.
        similarity_threshold: Threshold for considering two slides as similar (0 to 1).

    Returns:
        A list of unique slides.
    """
    from skimage.metrics import structural_similarity as ssim
    
    unique_slides = []
    seen_texts = set()

    for i, slide in enumerate(slides):
        is_duplicate = False

        # Check for duplicate OCR text
        if slide['text_ocr'] in seen_texts:
            continue

        for unique_slide in unique_slides:
            # Compare visual similarity
            img1 = cv2.imread(slide['source'], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(unique_slide['source'], cv2.IMREAD_GRAYSCALE)

            if img1 is not None and img2 is not None:
                score, _ = ssim(img1, img2, full=True)
                if score >= similarity_threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_slides.append(slide)
            seen_texts.add(slide['text_ocr'])

    return unique_slides


def extract_keyframes_from_video(target, output_json, thumb_dir):
    lo_size = (360, 202)
    hi_size = (1920, 1080)

    if thumb_dir is None:
        thumb_dir = tempfile.mkdtemp()

    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir)

    lo_dir = os.path.join(thumb_dir, 'lo')
    if not os.path.exists(lo_dir):
        os.makedirs(lo_dir)

    hi_dir = os.path.join(thumb_dir, 'hi')
    if not os.path.exists(hi_dir):
        os.makedirs(hi_dir)

    print("Extracting thumbnails...")
    extract_thumbnails(target, lo_dir, lo_size)

    print("Calculating delta images...")
    _, candidates, sse = get_delta_images(lo_dir, np.zeros(len(glob.glob(os.path.join(lo_dir, 'thumb-*.jpg'))), dtype=bool))
    print(f"Candidates: {candidates}")

    to_select = np.arange(len(glob.glob(os.path.join(lo_dir, 'thumb-*.jpg'))))
    print(f"Frames selected for slide detection: {to_select}")

    if len(to_select) == 0:
        print("No frames selected for slide detection. Exiting.")
        return

    print("Reconstructing slide deck...")
    sequence = max_likelihood_sequence(sse[to_select, :])
    print(f"Slide sequence: {sequence}")

    full_sequence = -np.ones(sse.shape[0])
    full_sequence[to_select] = sequence

    last_num = -2
    latest_slide = {'start_index': 0}
    slides = []

    # Get video duration for accurate timestamp calculation
    info = get_video_info(target)
    video_duration = float(info['duration'])
    total_frames = len(full_sequence)
    time_per_frame = video_duration / total_frames

    for i, num in tqdm(enumerate(full_sequence), desc="Processing slides"):
        if num != last_num:
            if last_num != -2:
                slides.append(latest_slide)

            latest_slide = {
                'type': 'slide',
                'start_time': to_timestamp(i * time_per_frame),  # Adjusted timestamp calculation
                'start_index': i,
                'offset': i * time_per_frame,  # Adjusted offset based on time
                'source': os.path.join(hi_dir, f"thumb-{i + 1:04}.png")  # Add source for all slides
            }
            last_num = num

    if last_num != -2:
        slides.append(latest_slide)

    # Ensure all slides are included in the sequence
    for i in tqdm(range(len(full_sequence)), desc="Including all slides"):
        if full_sequence[i] != -1:  # Include all slides, not just canonical ones
            slides.append({
                'type': 'slide',
                'start_time': to_timestamp(i * time_per_frame),  # Adjusted timestamp calculation
                'start_index': i,
                'offset': i * time_per_frame,
                'source': os.path.join(hi_dir, f"thumb-{i + 1:04}.png")
            })

    print(f"Detected slides: {slides}")

    for el in slides:
        if el['type'] == 'slide':
            el['source'] = os.path.join(hi_dir, f"thumb-{int(el['offset'] / time_per_frame) + 1:04}.png")

    offsets = [int(slide['offset'] / time_per_frame) for slide in slides if slide['type'] == 'slide']
    print(f"Extracting high-resolution frames for offsets: {offsets}")
    extract_frames(target, hi_dir, hi_size, offsets)

    info = {'pip_location': [],
            'sequence': slides}

    info['crop'] = extract_crop(info)
    print(f"Crop information: {info['crop']}")

    print(f"Found {len(slides)} canonical slides")

    unique_texts = set()
    unique_slides = []

    for el in tqdm(info['sequence'], desc="Processing OCR for slides"):
        if el['type'] == 'slide':
            im = cv2.imread(el['source'])
            if im is None:
                print(f"Warning: Could not read image {el['source']}, skipping OCR for this slide.")
                continue
            d = pytesseract.image_to_data(im, output_type=Output.DICT)
            text_ocr = ' '.join([d['text'][i] for i in range(len(d['text'])) if int(d['conf'][i]) > 80 and d['text'][i].strip()])

            if text_ocr not in unique_texts:
                unique_texts.add(text_ocr)
                el['text_ocr'] = text_ocr
                unique_slides.append(el)

    print("Deduplicating slides...")
    unique_slides = deduplicate_slides(unique_slides)
    print(f"Reduced slides from {len(info['sequence'])} to {len(unique_slides)}")

    info['sequence'] = unique_slides

    # Create Selected folder and copy selected images
    selected_dir = os.path.join(os.path.dirname(output_json), 'Selected')
    if not os.path.exists(selected_dir):
        os.makedirs(selected_dir)
    import shutil
    for el in unique_slides:
        if el['type'] == 'slide':
            src_img = el['source']
            img_name = os.path.basename(src_img)
            dst_img = os.path.join(selected_dir, img_name)
            shutil.copy2(src_img, dst_img)
            el['selected_image'] = dst_img  # Optionally add this path to JSON

    print("Writing output to JSON file...")
    with open(output_json, 'w') as f:
        json.dump(info, f, indent=2)
    print("JSON file created successfully.")
    # print json file path
    return output_json
    


if __name__ == "__main__":
    desc = """Extract key slides from video.
    
Extract key slides from video presenting a slide deck. The video could be a 
recording from Zoom or Google Meet, for example. The script extracts thumbnails
from the video and outputs a JSON file with timings for key slides. 
""" 
    target = r"D:\Video_Analysis\Video-Slide-Detector\Example_Video\Raymond James.mp4"
    
    # extract the video name from the path of target
    video_name = os.path.basename(target).split('.')[0]
    
    # create the output directory
    output_dir = os.path.join(os.path.dirname(target),'Output', video_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # create the output json file
    output_json = os.path.join(output_dir, f"{video_name}.json")
    
    # create the output directory for the thumbnails
    thumb_dir = os.path.join(output_dir, 'thumbnails')
    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir)
        
    extract_keyframes_from_video(target, output_json, thumb_dir)
