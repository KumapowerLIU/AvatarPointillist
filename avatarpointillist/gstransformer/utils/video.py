# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import numpy as np
import torch
import imageio
from typing import Tuple
import cv2

def images_to_video(images, output_path, fps, gradio_codec: bool, verbose=False):
    import imageio
    # images: torch.tensor (T, C, H, W), 0-1  or numpy: (T, H, W, 3) 0-255
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        if isinstance(images, torch.Tensor):
            frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
                f"Frame shape mismatch: {frame.shape} vs {images.shape}"
            assert frame.min() >= 0 and frame.max() <= 255, \
                f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        else:
            frame = images[i]
        frames.append(frame)
    frames = np.stack(frames)
    if gradio_codec:
        imageio.mimwrite(output_path, frames, fps=fps, quality=10)
    else:
        # imageio.mimwrite(output_path, frames, fps=fps, codec='mpeg4', quality=10)
        imageio.mimwrite(output_path, frames, fps=fps, quality=10)

    if verbose:
        print(f"Using gradio codec option {gradio_codec}")
        print(f"Saved video to {output_path}")


def save_images2video(img_lst, v_pth, fps):
    import moviepy.editor as mpy
    # Convert the list of NumPy arrays to a list of ImageClip objects
    clips = [mpy.ImageClip(img).set_duration(0.1) for img in img_lst]  # 0.1 seconds per frame

    # Concatenate the ImageClips into a single VideoClip
    video = mpy.concatenate_videoclips(clips, method="compose")

    # Write the VideoClip to a file
    video.write_videofile(v_pth, fps=fps)  # setting fps to 10 as example
    print("save video to:", v_pth)


def add_audio_to_video(video_path, out_path, audio_path):
    from moviepy.editor import VideoFileClip, AudioFileClip
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    video_clip_with_audio = video_clip.set_audio(audio_clip)
    video_clip_with_audio.write_videofile(out_path, codec='libx264', audio_codec='aac')
    print(f"Audio added successfully at {out_path}")


def save_imgs_2_video(img_lst, v_pth, fps=None):
    import numpy as np
    from moviepy.editor import ImageSequenceClip

    # 1. 检查并设置默认帧率，这部分是正确的
    if fps is None:
        fps = 25
    elif not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"[save_imgs_2_video] Invalid fps value: {fps}")

    images = [image.astype(np.uint8) for image in img_lst]

    # ✅ 修正 1：使用 fps 变量，而不是写死的 25
    clip = ImageSequenceClip(images, fps=fps)

    # ✅ 修正 2 & 3：使用 fps 变量，并修正 codec 的拼写
    clip.write_videofile(v_pth, codec='libx264', fps=fps)
    
    print(f"✅ Video saved successfully at {v_pth} with fps={fps}")



def save_imgs_2_video_imageio(img_lst, v_pth, fps=None):
    """
    使用 imageio (FFmpeg) 將圖像列表保存為高品質影片。
    """
    if fps is None:
        fps = 25

    # ✅ 使用 imageio.get_writer 創建一個寫入器
    # codec='libx264' -> 指定使用 H.264 編碼器
    # quality=8 -> 設置品質參數 (CRF)，數值越小品質越高，10是無損，8是很高品質
    # bitrate='20M' -> 或者直接指定比特率，例如 '20M' 代表 20Mbps，足以應對高清影片
    # 選擇 quality 或 bitrate 其中一個即可

    if isinstance(img_lst, np.ndarray):
        rgb_frames_list = [frame for frame in img_lst]
    else:
        rgb_frames_list = img_lst # 如果它本來就是列表，則不變
    if not rgb_frames_list:
        print("⚠️ 警告：圖像列表為空，無法創建影片。")
        return
    writer = imageio.get_writer(
        v_pth, 
        fps=fps, 
        codec='libx264', 
        quality=8, # 0-10 的品質等級，8-10 接近無損
        # bitrate='20M' # 或者使用這個來指定比特率
        macro_block_size=1 # 處理某些解析度問題
    )

    for image in rgb_frames_list:
        # imageio 預期的是 RGB 格式，所以不需要轉換！
        writer.append_data(image.astype(np.uint8))

    writer.close()
    print(f"✅ 使用 imageio 成功保存影片於 {v_pth}，幀率為 {fps}")
def save_imgs_2_video_cv2(img_lst, v_pth, fps=None):
    """
    使用 OpenCV (cv2) 將 NumPy 圖像列表保存為影片。

    Args:
        img_lst (list): 一個包含 NumPy 陣列的列表，每個陣列代表一幀圖像。
                        圖像格式應為 RGB，數據類型為 uint8。
        v_pth (str): 輸出影片的保存路徑 (例如 'output.mp4')。
        fps (int or float, optional): 影片的幀率。如果為 None，則默認為 25。
    """
    # --- 1. 參數檢查與設置 ---
    # 檢查圖像列表是否為空
    if isinstance(img_lst, np.ndarray):
        rgb_frames_list = [frame for frame in img_lst]
    else:
        rgb_frames_list = img_lst # 如果它本來就是列表，則不變
    if not rgb_frames_list:
        print("⚠️ 警告：圖像列表為空，無法創建影片。")
        return

    # 檢查並設置默認幀率，這部分邏輯保持不變
    if fps is None:
        fps = 25
    elif not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"[save_imgs_2_video_cv2] 無效的 fps 值: {fps}")

    # --- 2. 獲取影片尺寸並定義編碼器 ---
    # 從第一張圖片獲取高度、寬度
    # 假設所有圖片的尺寸都相同
    height, width, layers = rgb_frames_list[0].shape
    size = (width, height)

    # 定義影片編碼器 (Codec)。
    # 'mp4v' 是 .mp4 格式常用的 FourCC 編碼。
    # 其他選項如 'X264' (如果系統支持)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    # --- 3. 創建 VideoWriter 物件 ---
    writer = cv2.VideoWriter(v_pth, fourcc, fps, size)

    # --- 4. 逐幀寫入影片 ---
    for image in rgb_frames_list:
        # ⚠️ 關鍵步驟：OpenCV 處理顏色通道的順序是 BGR，而不是 RGB。
        # 由於你的輸入圖像很可能是 RGB 格式，我們需要先將其轉換為 BGR。
        image_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(image_bgr)

    # --- 5. 釋放資源 ---
    # 這是非常重要的一步，確保影片文件被正確地保存和關閉。
    writer.release()
    
    print(f"✅ 使用 OpenCV 成功保存影片於 {v_pth}，幀率為 {fps}")

# --- 如何使用 ---
# 假設你有一個名為 `rgb_frames` 的圖像列表
# rgb_frames = [...] # 列表中的每個元素都是一個 (H, W, 3) 的 NumPy 陣列
# save_imgs_2_video_cv2(rgb_frames, "my_video_with_opencv.mp4", fps=30)

def reshape_video(rgb: np.ndarray, N_views: int, h_w_grid: Tuple[int]):
    """
    Args:
        rgb: (N, H, W, 3)
        N_views: int
        h_w: (H, W)
    """
    assert len(h_w_grid) == 2, f"h_w should be a tuple of (height, width), but got {h_w_grid}"
    h_grid, w_grid = h_w_grid
    assert N_views == h_grid * w_grid, f"N_views should be equal to h_grid * w_grid, but got {N_views} != {h_grid} * {w_grid}"
    
    N, H, W, C = rgb.shape
    N_frames = N // N_views
    rgb = rgb[:N_frames * N_views, ...]

    rgb = rgb.reshape(N_frames, N_views, H, W, C).reshape(N_frames, h_grid, w_grid, H, W, C)
    rgb = rgb.transpose(0, 1, 3, 2, 4, 5).reshape(N_frames, h_grid * H, w_grid * W, C)

    print(f"reshape output video from {N}x{H}x{W}x{C} to {N_frames}x{h_grid * H}x{w_grid * W}x{C}...")
    return rgb

def reshape_video_self(rgb: np.ndarray, N_views: int, h_w_grid: Tuple[int]):
    """
    Args:
        rgb: (N, H, W, 3)
        N_views: int
        h_w: (H, W)
    """
    assert len(h_w_grid) == 2, f"h_w should be a tuple of (height, width), but got {h_w_grid}"
    h_grid, w_grid = h_w_grid
    assert N_views == h_grid * w_grid, f"N_views should be equal to h_grid * w_grid, but got {N_views} != {h_grid} * {w_grid}"
    
    N, H, W, C = rgb.shape
    N_frames = N // N_views
    rgb = rgb[:N_frames * N_views, ...]

    rgb = rgb.reshape(N_frames, N_views, H, W, C) 
    out_list = []
    for i in range(N_views):
        out_list.append(rgb[:, i, :, :, :])
    return out_list

if __name__ == "__main__":
    from glob import glob
    clip_name = "clip1"
    ptn = f"./assets/sample_motion/export/{clip_name}/images/*.png"
    images_pths = glob(ptn)
    import cv2
    import numpy as np
    images = [cv2.imread(pth) for pth in images_pths]
    save_images2video(images, "./assets/sample_mption/export/{clip_name}/video.mp4", 25, True)
