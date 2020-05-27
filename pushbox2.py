
'''
模拟真实推箱子游戏
1、固定的人物、箱子、目标点位置。
2、地图形状不是矩形。但是暂时做成矩形。如下：1是人物，2是箱子，3是矩形. +是墙
+ + + + +   +
+ 1     +   +
+   2 2 +   + + +
+   2   +   + 3 +
+ + +   + + + 3 +
+ + +         3 +
+ +       +     +
+ +       + + + +
+ + + + + +

3、多个箱子，多个目标点。修改状态空间设计，所有方块的状态可以用绝对值。因为目标点是不可移动的。所以只计算人物和箱子的状态。
总共有3+1个方块，每个方块有25种状态，总共有25的4次方即390625种可能。很小。
4、reward。每推一个箱子到目标点，从上到下依次是+45,+35,+25， 把箱子推出目标点减去原来的reward，其他-1
5、每集结束条件，3个箱子全部到达目标点上，或者超过200步。
6、移动方法需要改变。 推动箱子前要判断箱子前面是否有障碍(边界或者另一个箱子)，
因此要增加hasObstacle函数，并且action函数需要传入3个箱子的list。
设置一个List，存储人物和箱子的可达状态。除了这些可达点，其他的位置都是墙。

'''
from time import sleep
import time
from matplotlib import style
import pickle
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
all_state = [(1, 1), (1, 2), (1, 3),
             (2, 1), (2, 2), (2, 3),
             (3, 1), (3, 2), (3, 3), (3, 7),
             (4, 3), (4, 7),
             (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
             (6, 2), (6, 3), (6, 4), (6, 6), (6, 7),
             (7, 2), (7, 3), (7, 4)]

# style.use("ggplot")


SIZE = 9
HM_EPISODES = 10000

MOVE_PENALTY = -1
CAN_NOT_MOVE_PENALTY = -300
# DISTANCE_REDUCE_REWARD = 1
# 其实这里的设计不太对，如果移除箱子，应该有惩罚。
DEST_1_REWARD = 25
DEST_2_REWARD = 50
DEST_3_REWARD = 75

epsilon = 0.9

EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY

SHOW_EVERY = 1000  # how often to play through env visually.

# start_q_table = "qtable-1590171988.pickle"  # None or Filename
start_q_table = None  # None or Filename


LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
BOX_N = 2  # box key in dict
DEST_N = 3  # dest key in dict
WALL_N = 4
ROAD_N = 5

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255),
     4: (0, 0, 0),
     5: (255, 255, 255)}

# 8 小方块。用于定义玩家、箱子和目标点, 墙


def has_box_in_front(x, y, choice, box_list):
    if choice == 0:
        y = y-1
    elif choice == 1:
        y = y+1
    elif choice == 2:
        x = x-1
    elif choice == 3:
        x = x+1
    for box in box_list:
        if box.position() == (x, y):
            return box
    return None


def has_wall_in_front(x, y, choice):
    if choice == 0:
        y = y-1
    elif choice == 1:
        y = y+1
    elif choice == 2:
        x = x-1
    elif choice == 3:
        x = x+1
    for pos in all_state:
        if pos == (x, y):
            return True
    return False


def move(player, box, choice):
    if choice == 0:
        player.y = player.y - 1
        if box:
            box.y = box.y - 1
    elif choice == 1:
        player.y = player.y + 1
        if box:
            box.y = box.y + 1
    elif choice == 2:
        player.x = player.x - 1
        if box:
            box.x = box.x - 1
    elif choice == 3:
        player.x = player.x + 1
        if box:
            box.x = box.x + 1


class Blob:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice, box_list):
        '''
        Gives us 4 total movement options. (0,1,2,3). 分别为上下左右
        '''
        box = has_box_in_front(self.x, self.y, choice, box_list)
        if box:
            # 判断box的前面还有没有其他box或者墙
            res = has_box_in_front(box.x, box.y, choice, box_list)
            if not res:
                res = has_wall_in_front(box.x, box.y, choice)
                if not res:
                    # box和人物都往该方向移动一格
                    move(player, box, choice)
        else:
            # 没有box
            res = has_wall_in_front(self.x, self.y, choice)
            if not res:
                # 人物往该方向移动一格
                move(player, None, choice)

    def position(self):
        return self.x, self.y


# 从all_state表里取出来赋值
if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for player in all_state:  # 玩家
        for box1 in all_state:
            for box2 in all_state:
                for box3 in all_state:
                    q_table[(player, box1, box2, box3)] = [
                        np.random.uniform(-5, 0) for i in range(4)]


else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

# print(q_table)


def take_action(player, action, box_list, dest_list):
    # 这里要把箱子传进去。因为玩家的移动可能会影响箱子的位置。
    player.action(action, box_list)
    # 判断3个箱子有多少个在dest中,然后计算总的reward
    count = 0
    for box in box_list:
        for dest in dest_list:
            if box.position() == dest.position():
                count += 1
    if count > 0:
        reward = count * DEST_1_REWARD
    else:
        # 其他情况都-1
        reward = MOVE_PENALTY

    return reward


# TODO 这个很复杂，有3个box
def calc_q_value(reward, player, box_list):
    new_pos = (player.position(), box_list[0].position(),
               box_list[1].position(), box_list[2].position())
    max_future_q = np.max(q_table[new_pos])

    # 26 把当前相对位置状态和action下对应的收益qvalue取了出来。
    # TODO 这个要思考一下是不是new_pos
    current_q = q_table[new_pos][action]

    new_q = (1 - LEARNING_RATE) * current_q + \
        LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    return new_q


def show_movement(player, box_list, dest_list):
    # 前面二维是x，y，第三的维度是颜色的索引。其实直接传颜色进去也行吧。本来颜色。
    # 用这种方式定义了一个环境，x值，y值，以及格子颜色
    # starts an rbg of our size
    # 首先把所有区域设置成墙
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    # 然后修改可达点的颜色
    for x, y in all_state:
        env[x, y] = d[ROAD_N]

    # 然后再修改玩家、box、dest的颜色
    # sets the player tile to blue
    env[player.x][player.y] = d[PLAYER_N]
    # sets the box location tile to green color
    for box in box_list:
        env[box.x][box.y] = d[BOX_N]

    # sets the dest location to red
    for dest in dest_list:
        env[dest.x][dest.y] = d[DEST_N]

    # 30 构建image的一种方法，传入一个二维数组。数组的长宽是image的长宽，数组的值是image的颜色。
    # reading to rgb. Apparently. Even tho color definitions are bgr. ???
    # 上面构建env主要是为了方便下面生成cv图像
    img = Image.fromarray(env, 'RGB')
    # 31 image进行拉伸，不然太小了。
    # resizing so we can see our agent in all its glory.
    # print(img)
    img = img.resize((300, 300))
    # print(img)
    # 31 又把这个image变成了数组？没看懂，固定套路？
    cv2.imshow("image", np.array(img))  # show it!
    # crummy code to hang at the end if we reach abrupt end for good reasons or not.
    if reward == DEST_3_REWARD:
        # 32 ord是什么意思？ 点q键。返回q的unicode字符编码。
        if cv2.waitKey(500) & 0xFF == ord('q'):
            return False
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    sleep(0.1)


# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example


episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob(1, 1)
    box_list = [Blob(2, 2), Blob(2, 3), Blob(3, 2)]
    dest_list = [Blob(7, 3), Blob(7, 4), Blob(7, 5)]
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(
            f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    last_distance = None
    for i in range(200):
        # 这里就需要修改了，不用通过相对位置计算reward。 只有到达光标点计算reward

        # 通过当前的位置计算出最大q-value的行动
        if np.random.random() > epsilon:
            action = np.argmax(q_table[player.position(),
                                       box_list[0].position(
            ), box_list[1].position(),
                box_list[2].position()])
        else:
            action = np.random.randint(0, 4)

        # Take the action!
        reward = take_action(player, action, box_list, dest_list)

        # NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_q = calc_q_value(reward, player, box_list)
        q_table[obs][action] = new_q

        if show:
            show_movement(player, box_list, dest_list)

        # 用于计算平均reward
        episode_reward += reward

        # 33 结束.碰到敌人或者碰到食物。 推箱子中也是，箱子碰到目标点就结束。
        if reward == DEST_3_REWARD or reward == CAN_NOT_MOVE_PENALTY:
            break

    # print(episode_reward)
    # 34 rewards会呈现怎样的规律？越来越高
    episode_rewards.append(episode_reward)
    # 35 自主探索的概率要不断的降低。有意思。可能实际应用中都是这样吧。当一个人越来越有经验后，就越不需要自主探索。
    epsilon *= EPS_DECAY

# ?36 这个平均数是什么意思?
moving_avg = np.convolve(episode_rewards, np.ones(
    (SHOW_EVERY,))/SHOW_EVERY, mode='valid')
print(moving_avg)

# ？37 画最后的统计图。计算pingjunreward？
plt.plot([i for i in range(len(moving_avg))], moving_avg)
# 38 纵坐标是reward，横坐标是集
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

# 38 最后会把q-table存起来。action4个。state好像很多，100*100=1万种情况？ 错了 总共16万种。
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
