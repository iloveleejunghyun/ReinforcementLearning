
'''
以下，将从躲避敌人找食物的程序变成推箱子的程序。
先设计一个最简单的推箱子第一关。正方形场地，只有1个箱子，一个目标点。
代码需要改变的地方:
1、敌人改成目标点，食物改成箱子。
2、任务改变。原本任务，躲避敌人，找到食物。
    现任务：将箱子推到目标点。
    任务的改变真正影响的是reward，
    原本任务: 1 碰到敌人，reward-300；2 碰到食物，+25；3 其他，-1
    现任务：1 箱子到达目标点，reward+25； 2 箱子离目标点距离减少，+1；
            3 箱子到达推不动的位置，-300； 4 其他 -1
3、动作不变，上下左右，q-table也可以不变。
    状态要改变:原状态：玩家与食物的相对位置，玩家与敌人的相对位置。
    现状态，玩家与箱子的相对位置，箱子与目标点的相对位置。

4、行动。
    如果玩家前面是箱子，则会推动箱子前进。如果箱子前面是墙壁，则箱子和玩家都不会动。
'''


# style.use("ggplot")

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
from time import sleep

SIZE = 10
HM_EPISODES = 100000

MOVE_PENALTY = -1
CAN_NOT_MOVE_PENALTY = -300
DISTANCE_REDUCE_REWARD = 1
DISTANCE_0_REWARD = 25


epsilon = 0.0

EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY

SHOW_EVERY = 10000  # how often to play through env visually.

start_q_table = "qtable-1590815101.pickle"  # None or Filename
# start_q_table = None  # None or Filename


LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
BOX_N = 2  # box key in dict
DEST_N = 3  # dest key in dict

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

# 8 小方块。用于定义玩家、箱子和目标点


class Blob:
    def __init__(self, edge=True):
        if edge:
            self.x = np.random.randint(0, SIZE)
            self.y = np.random.randint(0, SIZE)
        else:
            self.x = np.random.randint(1, SIZE-1)
            self.y = np.random.randint(1, SIZE-1)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice, box):
        '''
        Gives us 4 total movement options. (0,1,2,3). 分别为上下左右
        '''

        if choice == 0:
            # 上
            if box.x == self.x and box.y == self.y-1:
                # box在玩家上面一格
                if box.y == 0:
                    # box在边界，推不动
                    return
                else:
                    # box和player同时向上移动一格
                    box.y = box.y-1
                    self.y = self.y-1
            else:

                if self.y == 0:
                    return
                else:
                    # 只有玩家上移动一格
                    self.y = self.y-1
            return
        elif choice == 1:
            # 下
            if box.x == self.x and box.y == self.y+1:
                # box在玩家下面一格
                if box.y == SIZE-1:
                    # box在边界，推不动
                    return
                else:
                    # box和player同时向下移动一格
                    box.y = box.y+1
                    self.y = self.y+1
            else:

                if self.y == SIZE-1:
                    return
                else:
                    # 只有玩家下移动一格
                    self.y = self.y+1
            return

        elif choice == 2:
            # 左
            if box.x == self.x-1 and box.y == self.y:
                # box在玩家左面一格
                if box.x == 0:
                    # box在边界，推不动
                    return
                else:
                    # box和player同时向左移动一格
                    box.x = box.x-1
                    self.x = self.x-1
            else:

                if self.x == 0:
                    return
                else:
                    # 只有玩家左移动一格
                    self.x = self.x-1
            return
        elif choice == 3:
            # 右
            if box.x == self.x+1 and box.y == self.y:
                # box在玩家右面一格
                if box.x == SIZE-1:
                    # box在边界，推不动
                    return
                else:
                    # box和player同时向右移动一格
                    box.x = box.x+1
                    self.x = self.x+1
            else:

                if self.x == SIZE-1:
                    return
                else:
                    # 只有玩家右移动一格
                    self.x = self.x+1
            return

    # 这个是为了拿来随机移动，暂时不用。如果箱子食物能够随机移动的话，就会变得更加复杂。可能以上算法不能让q-table收敛。
    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

       # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                for iiii in range(-SIZE+1, SIZE):
                    q_table[((i, ii), (iii, iiii))] = [
                        np.random.uniform(-5, 0) for i in range(4)]


else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

# print(q_table)


def take_action(player, action, box, dest):
    # 这里要把箱子传进去。因为玩家的移动可能会影响箱子的位置。
    player.action(action, box)
    # 这里要修改为距离的增加或者减少
    cur_distance = (box.x - dest.x) ** 2 + (box.y - dest.y) ** 2
    if cur_distance == 0:
        reward = DISTANCE_0_REWARD
    elif last_distance > cur_distance:
        reward = DISTANCE_REDUCE_REWARD
    # TODO 这里还要增加箱子移动到角落卡住的惩罚。
    else:
        reward = MOVE_PENALTY
    return reward


def calc_q_value(reward, player, box, dest, old_obs):
    new_obs = (player-box, box-dest)
    max_future_q = np.max(q_table[new_obs])
    # 26 把当前相对位置状态和action下对应的收益qvalue取了出来。
    current_q = q_table[old_obs][action]

    # q-value最大值只能是箱子到达目标点的收益
    # if reward == DISTANCE_0_REWARD:
    #     new_q = DISTANCE_0_REWARD
    # else:
    # 其他情况按照公式计算
    # 这样应该会导致new_q上限变得更高
    new_q = (1 - LEARNING_RATE) * current_q + \
        LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    return new_q


def show_movement(player, box, dest):
    # 28 传入了3元组？，输出是个3维数组。前面二维是x，y，第三的维度？是颜色的索引。其实直接传颜色进去也行吧。本来颜色。
    # 用这种方式定义了一个环境，x值，y值，以及格子颜色
    # starts an rbg of our size
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    # 29 设置颜色。数组中的每个位置的值是颜色。
    # sets the box location tile to green color
    env[box.x][box.y] = d[BOX_N]
    # sets the player tile to blue
    env[player.x][player.y] = d[PLAYER_N]
    # sets the dest location to red
    env[dest.x][dest.y] = d[DEST_N]
    # 30 构建image的一种方法，传入一个二维数组。数组的长宽是image的长宽，数组的值是image的颜色。
    # reading to rgb. Apparently. Even tho color definitions are bgr. ???
    # 上面构建env主要是为了方便下面生成cv图像
    img = Image.fromarray(env, 'RGB')
    # 31 image进行拉伸，不然太小了。
    # resizing so we can see our agent in all its glory.
    # print(img)
    img = img.resize((300, 300), Image.NEAREST)
    # print(img)
    # 31 又把这个image变成了数组？没看懂，固定套路？
    cv2.imshow("image", np.array(img))  # show it!
    # crummy code to hang at the end if we reach abrupt end for good reasons or not.
    if reward == DISTANCE_0_REWARD or reward == CAN_NOT_MOVE_PENALTY:
        # 32 ord是什么意思？ 点q键。返回q的unicode字符编码。
        if cv2.waitKey(500) & 0xFF == ord('q'):
            return False
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    # sleep(0.1)


# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example


episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    box = Blob(False)
    dest = Blob()
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

        obs = (player-box, box-dest)
        if last_distance == None:
            last_distance = (box.x - dest.x) ** 2 + \
                (box.y - dest.y) ** 2  # 上一次的距离初始化为当前第一次距离

        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Take the action!
        reward = take_action(player, action, box, dest)

        #### MAYBE ###
        # enemy.move()
        # food.move()
        ##############

        # NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_q = calc_q_value(reward, player, box, dest, obs)
        q_table[obs][action] = new_q

        if show:
            show_movement(player, box, dest)

        # 32 episode_reward是指每一集的总reward, 意义何在？ 后面要计算平均值，用于统计是否进步。
        episode_reward += reward

        # 33 结束.碰到敌人或者碰到食物。 推箱子中也是，箱子碰到目标点就结束。
        if reward == DISTANCE_0_REWARD or reward == CAN_NOT_MOVE_PENALTY:
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
