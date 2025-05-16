import wikipedia
import subprocess

def search_contents(text) :
    
    wikipedia.set_lang('ja')

    try:
        wp = wikipedia.page(text)
        return wp.summary
    except wikipedia.exceptions.PageError:
        return "該当するページが見つかりません。"

def search_contents2(text) :
    
    wikipedia.set_lang('ja')

    try:
        wp = wikipedia.page(text)
        return wp.content
    except wikipedia.exceptions.PageError:
        return "該当するページが見つかりません。"

def run_process1(content,model):
    prompt = """あなたはクイズ作成の達人です。
    以下のフォーマットをもとにし、wikipediaの情報から
    日本語でクイズを作成してください。

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

    【クイズ作成のルール】  
    - Wikipediaの内容をそのまま要約しないでください。  
    - 直接クイズを作成し、要約や説明文は出力しないでください。    
    - 各問題の選択肢には、一つだけ明らかに間違った選択肢を含めてください。  
    - 出題は日本語で行い、平易な表現を使ってください。 
    
    クイズを作成する対象:【""" + content + "】"

    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    #プロンプトを送信して結果を取得
    stdout, stderr = process.communicate(prompt)

    # 出力を表示
    print("Response:", stdout)
    if stderr:
        print("Error:", stderr)

def run_process2(content,model):
    prompt = """次のWikipediaの情報をもとに、日本語で4択クイズを作成してください。  

手順:  
1. 重要なポイントを3つ抽出してください。  
2. 各ポイントについて「○○とは何か？」「○○の特徴は？」などのクイズを考えてください。  
3. それぞれのクイズに対して、選択肢（A, B, C, Dの4択）を作成してください。  
4. 正解と、その理由を簡潔に説明してください。  

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
    
    クイズを作成する対象:【""" + content + "】"

    process = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    #プロンプトを送信して結果を取得
    stdout, stderr = process.communicate(prompt)

    # 出力を表示
    print("Response:", stdout)
    if stderr:
        print("Error:", stderr)
target = 'ヤンバルクイナ'
model = 'deepseek-r1:14b'

content = search_contents(target)
run_process1(content,model)

