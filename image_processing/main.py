import cv2

#画像パス
path = "image/qoobee.png"

#画像読み込み
img = cv2.imread(path)
img_g = cv2.imread(path,0)

#画像情報
def img_info(img):
    print(img)

#画像の表示
def img_show(img):
    cv2.imshow("image",img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

#グレースケール画像の表示
def img_show_gray(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("image",img_gray)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

#図形足し
def img_add(img):
    #ラインを書く
    #cv2.line(入力画像,左上の座標,右下の座標,BGR値,太さ)
    cv2.line(img,(50, 10), (125, 600), (255, 0, 0),2)
    #長方形を書く
    #cv2.rectangle(入力画像,左上の座標,右下の座標,BGR値,太さ
    cv2.rectangle(img,(100,25),(300,150),(0,255,0),3)
    #円を書く
    #cv2.circle(入力画像,円の中心座標,半径,BGR値,太さ)
    cv2.circle(img,(80,120),20,(0,0,250),-1)
    #楕円を書く
    #cv2.ellipse(入力画像,中心座標,長軸,短軸,楕円の傾き具合,楕円を描画する始まりの角度,終わりの角度,BGR値,太さ)
    cv2.ellipse(img,(30,100),(100,50),20,0,360,(255,0,0),3)
    #文字を書く
    #cv2.putText(入力画像,記述する文字,文字の座標,フォント,フォントスケール,BGR値,文字の太さ,線のタイプ)
    cv2.putText(img, 'Ryudai!', (70, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (200, 145, 124
                                                                        ), lineType=cv2.LINE_AA)
    img_show(img)

#キャニーのエッジ検出
def canny_edge(img):
    img_canny = cv2.Canny(img,100,100)
    img_show(img_canny)

#img_info(img)
#img_show(img)
#img_show_gray(img)
img_add(img)
#canny_edge(img_g)