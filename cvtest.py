
import cv2
from PIL import Image
import numpy as np
import time

SIZE = 10

PLAYER_N = 1  # player key in dict
BOX_N = 3  # box key in dict
DEST_N = 2  # dest key in dict
WALL_N = 4
ROAD_N = 5

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255),
     4: (0, 0, 0),
     5: (255, 255, 255)}


def show():
    # 前面二维是x，y，第三的维度是颜色的索引。其实直接传颜色进去也行吧。本来颜色。
    # 用这种方式定义了一个环境，x值，y值，以及格子颜色
    # starts an rbg of our size
    # 首先把所有区域设置成墙
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    # 然后再修改玩家、box、dest的颜色
    # sets the dest location to red

    env[0][0] = d[DEST_N]

    # sets the player tile to blue
    env[1][0] = d[PLAYER_N]

    # sets the box location tile to green color

    env[2][0] = d[BOX_N]

    # 30 构建image的一种方法，传入一个二维数组。数组的长宽是image的长宽，数组的值是image的颜色。
    # reading to rgb. Apparently. Even tho color definitions are bgr. ???
    # 上面构建env主要是为了方便下面生成cv图像
    img = Image.fromarray(env, 'RGB')
    # 31 image进行拉伸，不然太小了。
    # resizing so we can see our agent in all its glory.
    # print(img)
    img = img.resize((300, 300), Image.NEAREST)
    img.show()
    # print(img)
    # 31 又把这个image变成了数组？没看懂，固定套路？
    cv2.imshow("image", np.array(img))  # show it!
    # crummy code to hang at the end if we reach abrupt end for good reasons or not.

    # 32 ord是什么意思？ 点q键。返回q的unicode字符编码。
    if cv2.waitKey(5000) & 0xFF == ord('q'):
        return False


show()
