import imageio.v2 as imageio
import os

def make_video(image_folder: str, output_path: str, fps: int = 30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    if not images:
        print("No images found in the specified folder.")
        return
    images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    with imageio.get_writer(output_path, fps=fps, format='FFMPEG') as video:
        for filename in images:
            file_path = os.path.join(image_folder, filename)
            image = imageio.imread(file_path)
            video.append_data(image)