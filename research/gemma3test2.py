import time
import requests
import base64
from PIL import Image
import io
from typing import Optional, List

def load_and_resize_image(image_path: str, max_size: int = 512) -> str:
    """
    画像を読み込み、アスペクト比を維持しながら最大辺をmax_sizeにリサイズ。
    リサイズ後、Base64エンコードした文字列を返す。
    """
    image = Image.open(image_path)
    w, h = image.size

    if max(w, h) > max_size:
        if w >= h:
            new_w = max_size
            new_h = int(h * (max_size / w))
        else:
            new_h = max_size
            new_w = int(w * (max_size / h))
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def generate_text_with_images(prompt: str, image_paths: Optional[List[str]] = None, max_new_tokens: int = 300, max_size: int = 512) -> str:
    """
    複数画像に対応したプロンプトをGemma3:12bモデルに投げて、応答を得る。
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:12b",
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.5,
        "top_p": 0.95,
        "stream": False
    }

    if image_paths:
        payload["images"] = [load_and_resize_image(p, max_size=max_size) for p in image_paths]

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")

if __name__ == "__main__":
    # 画像間の比較プロンプト
    prompt = "これら2枚の画像を比較し、共通点と相違点を説明してください。"

    # 比較したい画像のパス
    image_paths = [
        "/Users/adachiharuto/prog/research/シリアルミルク.jpg",
        "/Users/adachiharuto/prog/research/牛丼.jpg"  # 比較対象の画像を指定
    ]

    result = generate_text_with_images(prompt, image_paths=image_paths)
    print(result)
