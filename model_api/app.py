from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

def load_onnx_model(onnx_path):
    try:
        sess = ort.InferenceSession(onnx_path)
        print(f"成功加载模型: {onnx_path}")
        return sess
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

# 加载模型
model1 = load_onnx_model(os.path.join(os.path.dirname(__file__),"model_weights","best.onnx"))
model2 = load_onnx_model(os.path.join(os.path.dirname(__file__),"model_weights","cnn.onnx"))
if not model1 or not model2:
    exit(1)  # 退出程序


def preprocess_image(image_bytes):
    """将上传的图片转换为模型需要的格式"""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((640, 640))  # YOLOv5输入尺寸
    image_np = np.array(image).transpose(2, 0, 1)  # HWC -> CHW
    return image_np.astype(np.float32) / 255.0  # 归一化


@app.route('/detect', methods=['POST'])
def detect():
    # 1. 接收小程序上传的图片
    file = request.files['image']
    img_data = file.read()

    # 2. 预处理
    input_tensor = preprocess_image(img_data)[np.newaxis, ...]  # 增加batch维度

    # 3. 运行YOLOv5模型
    outputs = model1.run(
        None,
        {"images": input_tensor}
    )

    # 4. 处理结果（示例：提取检测框）
    boxes = outputs[0][:, :, :4]  # 假设前4个值是坐标
    return jsonify(boxes.tolist())


@app.route('/classify', methods=['POST'])
def classify():
    # 类似detect的逻辑，使用cnn_session
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)