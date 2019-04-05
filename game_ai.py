from lib.Test_forai import *
from lib.game_cnn import *

'''
原來如果一個 python script 是被別的 python script 當成 module 來 import 的話
，那麼這個被 import 的 python script 的 __name__ 就會是那個 python script 的
名稱。而如果這個 python script 是直接被執行的話，__name__ 會是 __main__。
'''
if __name__ == '__main__':
  ai = AIGame()
  #ai.train_start()
  ai.ai_play_start()
pass
