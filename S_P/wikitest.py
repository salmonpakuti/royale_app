import wikipedia
import subprocess

def search_contents(text):
    wikipedia.set_lang('ja')
    try:
        wp = wikipedia.page(text)
        return wp.content
    except wikipedia.exceptions.PageError:
        return "該当するページが見つかりません。"

def summarize_content(content, model):
    prompt = """次のWikipediaの内容を300文字以内に要約してください：
    【内容】
    """ + content + """
    """
    
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(prompt)
    if stderr:
        print("Error:", stderr)
    print("要約:", stdout.strip())  # 要約を出力
    return stdout.strip()

def run_process1(content, model):
    prompt = """あなたはクイズ作成の達人です。以下の要約をもとに日本語でクイズを３問作成してください。
    
    【要約】
    """ + content + """
    
    【出力フォーマット】  
    1. 問題: (ここに問題文)  
    A) (選択肢1)  
    B) (選択肢2)  
    C) (選択肢3)  
    D) (選択肢4)  
    正解: (A, B, C, D のいずれか)  
    
    2. 問題: (ここに問題文)  
    A) (選択肢1)  
    B) (選択肢2)  
    C) (選択肢3)  
    D) (選択肢4)  
    正解: (A, B, C, D のいずれか)  
    
    3. 問題: (ここに問題文)  
    A) (選択肢1)  
    B) (選択肢2)  
    C) (選択肢3)  
    D) (選択肢4)  
    正解: (A, B, C, D のいずれか)
    """
    
    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(prompt)
    print("Response:", stdout)
    if stderr:
        print("Error:", stderr)

# 実行
model = 'gemma3:12b'
target = 'ヤンバルクイナ'

full_content = search_contents(target)
summary = summarize_content(full_content, model)
run_process1(summary, model)