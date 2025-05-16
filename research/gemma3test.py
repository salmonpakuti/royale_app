import time
import requests
import base64
from PIL import Image
import io
from typing import Optional

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

def generate_text(prompt: str, image_path: Optional[str] = None, max_new_tokens: int = 300, max_size: int = 512) -> str:
    """
    指定したプロンプトからGemma3:4bモデルでテキスト生成を実施する。
    画像パスが渡された場合、画像入力モードになり、リサイズ時のmax_sizeが反映される。
    """
    url = "http://localhost:11434/api/generate"  # Ollamaサーバーのエンドポイント
    payload = {
        "model": "gemma3:12b",
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.5,
        "top_p": 0.95,
        "stream": False
    }

    if image_path:
        encoded_image = load_and_resize_image(image_path, max_size=max_size)
        payload["images"] = [encoded_image]

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")

if __name__ == "__main__":
    prompt = "この画像に写っているオブジェクトを認識して教えてください"
    prompt2 = "Recognize the object in this image and tell me."
    image_path = "images/LINE_ALBUM_宜野湾市立博物館_240812_27.jpg"
    result= generate_text(prompt2, image_path=image_path)
    print(result)
