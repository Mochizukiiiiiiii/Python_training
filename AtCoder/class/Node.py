#幅優先探索

#クラスを宣言
class Node:
    #コンストラクタを宣言
    def __init__(self, index):
        #メソッドを定義
        self.index = index #Nodeの番号を定義
        self.nears = [] #隣接Nodeのリストを定義
        self.sign = False #探索済みかどうか定義

    def __repr__(self):
        return f'Node index:{self.index} Node nears:{self.nears} Node sign:{self.sign}'