import numpy as np
import os
import cv2
from utils import video_augmentation
from slr_network import SLRModel
import torch
from collections import OrderedDict
import utils
from flask import Flask, request, render_template_string
from decord import VideoReader, cpu
import warnings

warnings.filterwarnings("ignore")
VIDEO_FORMATS = [".MP4", ".mp4", ".avi", ".mov", ".mkv"]

# -----------------------------
# Helper Functions
# -----------------------------

def is_image_by_extension(file_path):
    _, file_extension = os.path.splitext(file_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    return file_extension.lower() in image_extensions

def load_video(video_path, max_frames_num=360):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    if total_frame_num > max_frames_num:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    else:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return [cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB) for tmp in spare_frames]

def run_inference(inputs, max_frames_num=360):
    img_list = []
    if isinstance(inputs, list):  # Multi-image case
        for x in inputs:
            if is_image_by_extension(x):
                img_list.append(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB))

    elif os.path.splitext(inputs)[-1] in VIDEO_FORMATS:  # Video case
        img_list = load_video(inputs, max_frames_num)
    else:
        raise ValueError("Unsupported input format")

    transform = video_augmentation.Compose([
        video_augmentation.CenterCrop(224),
        video_augmentation.Resize(1.0),
        video_augmentation.ToTensor(),
    ])
    vid, _ = transform(img_list, None, None)
    vid = vid.float() / 127.5 - 1
    vid = vid.unsqueeze(0)

    # Padding
    left_pad, last_stride, total_stride = 0, 1, 1
    kernel_sizes = ['K5', "P2", 'K5', "P2"]
    for ks in kernel_sizes:
        if ks[0] == 'K':
            left_pad = left_pad * last_stride
            left_pad += int((int(ks[1]) - 1) / 2)
        elif ks[0] == 'P':
            last_stride = int(ks[1])
            total_stride *= last_stride

    max_len = vid.size(1)
    video_length = torch.LongTensor([np.ceil(vid.size(1) / total_stride) * total_stride + 2 * left_pad])
    right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
    max_len = max_len + left_pad + right_pad
    vid = torch.cat(
        (
            vid[0, 0][None].expand(left_pad, -1, -1, -1),
            vid[0],
            vid[0, -1][None].expand(max_len - vid.size(1) - left_pad, -1, -1, -1),
        ),
        dim=0).unsqueeze(0)

    vid = device.data_to_device(vid)
    vid_lgt = device.data_to_device(video_length)
    print("the shape of the video is:", vid.shape)
    ret_dict = model(vid, vid_lgt, label=None, label_lgt=None)
    print("the detected sentence glosses:",ret_dict['recognized_sents'])
    return ret_dict['recognized_sents']

# -----------------------------
# Load Model (once at startup)
# -----------------------------

model_path = "./new_work_dir/baseline/_best_model.pt"
dataset = "phoenix2014-T"
dict_path = f'./preprocess/{dataset}/gloss_dict.npy'

gloss_dict = np.load(dict_path, allow_pickle=True).item()
device = utils.GpuDataParallel()
device.set_device(0)

model = SLRModel(
    num_classes=len(gloss_dict) + 1,
    c2d_type='resnet18',
    conv_type=2,
    use_bn=1,
    gloss_dict=gloss_dict,
    loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0,'Cu': 0.0005,'Cp': 0.0005},
)

state_dict = torch.load(model_path)['model_state_dict']
state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
model.load_state_dict(state_dict, strict=True)
model = model.to(device.output_device).cuda()
model.eval()


# -----------------------------
# Flask App
# -----------------------------

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Continuous Sign Language Recognition</title>
    </head>
    <body>
        <h2>Upload Video or Multiple Images for Inference</h2>
        <form action="/infer" method="post" enctype="multipart/form-data">
            <input type="file" name="files" multiple required>
            <br><br>
            <button type="submit">Upload & Run Inference</button>
        </form>
    </body>
    </html>
    """)

@app.route("/infer", methods=["POST"])
def infer():
    if "files" not in request.files:
        return "No files uploaded", 400

    files = request.files.getlist("files")
    file_paths = []
    for file in files:
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)
        file_paths.append(save_path)

    try:
        if len(file_paths) == 1 and os.path.splitext(file_paths[0])[-1] in VIDEO_FORMATS:
            results = run_inference(file_paths[0])
        else:
            results = run_inference(file_paths)

        # Limit previews to 5
        preview_files = file_paths[:5]
        previews_html = ""
        for f in preview_files:
            ext = os.path.splitext(f)[-1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                previews_html += f'<img src="/{f}" width="200"><br>'
            elif ext in VIDEO_FORMATS:
                previews_html += f'<video width="320" controls><source src="/{f}" type="video/mp4"></video><br>'

        return render_template_string(f"""
        <!DOCTYPE html>
        <html>
        <head><title>Results</title></head>
        <body>
            <h2>Inference Results</h2>
            <p><b>Recognized Sentences:</b> {results}</p>
            <h3>Uploaded Files Preview (showing {len(preview_files)} of {len(file_paths)})</h3>
            {previews_html}
            <br><a href="/">Upload More</a>
        </body>
        </html>
        """)
    except Exception as e:
        return f"Error: {str(e)}", 500

# Serve uploaded files
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)