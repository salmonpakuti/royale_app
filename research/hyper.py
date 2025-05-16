import time
import requests
import base64
from PIL import Image
import io
from typing import Optional

def load_and_resize_image(image_path: str, max_size: int) -> str:
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

def generate_text(prompt: str, encoded_image: Optional[str] = None, max_new_tokens: int = 300) -> str:
    """
    指定したプロンプトからGemma3:4bモデルでテキスト生成を実施する。
    画像のBase64エンコード済み文字列が渡された場合、画像入力モードとして利用する。
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
    
    if encoded_image:
        payload["images"] = [encoded_image]
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")

if __name__ == "__main__":
    max_sizes = [1024, 512, 256, 128, 64,32,16]
    prompt = """
    適宜プロンプトを変更
    """
    image_path = "/Users/adachiharuto/prog/research/家族.jpg"  # 実際の画像パスに変更してください
    num_trials = 3  # 各 max_size ごとの試行回数

    final_results = {}

    # 各 max_size ごとに画像処理は1回だけ実施し、試行ごとに実行時間と生成結果を記録
    for size in max_sizes:
        print(f"Processing max_size = {size}...")
        encoded_image = load_and_resize_image(image_path, max_size=size)
        trial_results = []
        for trial in range(1, num_trials + 1):
            print(f"  Trial {trial} for max_size = {size} started...")
            start_time = time.perf_counter()
            result = generate_text(prompt, encoded_image=encoded_image, max_new_tokens=300)
            result = result.replace('```json','').replace('```','')
            print(result)
            elapsed = time.perf_counter() - start_time
            trial_results.append((elapsed, result))
            print(f"  Trial {trial} for max_size = {size} finished in {elapsed:.2f}秒")
        final_results[size] = trial_results
        print(f"Completed processing for max_size = {size}\n")
    # 結果を指定のマークダウン形式で整形し、result.txt に書き出す
    output_lines = []
    output_lines.append("# 実験結果\n")
    for size in max_sizes:
        # 各 max_size ごとの平均実行時間を計算
        times = [trial[0] for trial in final_results[size]]
        avg_time = sum(times) / len(times)
        
        output_lines.append(f"<details>")
        output_lines.append(f"<summary>max_size={size} 生成結果（クリックで展開）</summary>")
        output_lines.append("\n```console")
        for trial, (elapsed, result) in enumerate(final_results[size], 1):
            output_lines.append(f"Trial {trial}: {elapsed:.2f}秒")
            output_lines.append(result)
            output_lines.append("")
        output_lines.append(f"平均実行時間: {avg_time:.2f}秒")
        output_lines.append("```")
        output_lines.append("</details>\n")
    
    final_output = "\n".join(output_lines)
    
    # result.txt に書き出し
    with open("result_json.txt", "w", encoding="utf-8") as f:
        f.write(final_output)
    
    print("結果は result.txt に書き出されました。")
