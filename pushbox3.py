
'''
模拟真实推箱子游戏
1、固定的人物、箱子、目标点位置。
2、地图形状不是矩形。但是暂时做成矩形。如下：1是人物，2是箱子，3是矩形. +是墙




+ + + +
+     +
+     + + + + + + + + + +
+         + +           +
+ 3 3 +         2 2 +   +
+ 3 3     + +       2   + + +
+ 3 3 +     + + 2 +   2     +
+ 3 3       +   1 2   2     +
+ 3 3 +     +   2   2       +
+   3       +   2   2   + + +
+     +     +       + + +
+     +         + + +
+ + + + + + + + +

3、多个箱子，多个目标点。修改状态空间设计，所有方块的状态可以用绝对值。因为目标点是不可移动的。所以只计算人物和箱子的状态。
'''
import random
from time import sleep
import time
from matplotlib import style
import pickle
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
dead_state = [(1, 1), (2, 1),
              (4, 3), (7, 3), (8, 3), (9, 3), (10, 3), (11, 3),
              (11, 4),
              (7, 5),
              (5, 6), (13, 6),
              (5, 7), (7, 7), (13, 7),
              (5, 8), (7, 8), (13, 8),
              (5, 9), (7, 9), (11, 9),
              (5, 10), (7, 10), (9, 10),
              (1, 11), (2, 11), (4, 11), (5, 11), (6, 11), (7, 11)
              ]

all_state = [(1, 1), (2, 1),
             (1, 2), (2, 2),
             (1, 3), (2, 3), (3, 3), (4, 3), (7,
                                              3), (8, 3), (9, 3), (10, 3), (11, 3),
             (1, 4), (2, 4), (4, 4), (5, 4), (6,
                                              4), (7, 4), (8, 4), (9, 4), (11, 4),
             (1, 5), (2, 5), (3, 5), (4, 5), (7, 5), (8,
                                                      5), (9, 5), (10, 5), (11, 5),
             (1, 6), (2, 6), (4, 6), (5, 6), (8,
                                              6), (10, 6), (11, 6), (12, 6), (13, 6),
             (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (7, 7), (8,
                                                              7), (9, 7), (10, 7), (11, 7), (12, 7), (13, 7),
             (1, 8), (2, 8), (4, 8), (5, 8), (7,
                                              8), (8, 8), (9, 8), (10, 8), (11, 8), (12, 8), (13, 8),
             (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (7,
                                                      9), (8, 9), (9, 9), (10, 9), (11, 9),
             (1, 10), (2, 10), (4, 10), (5, 10), (7, 10), (8, 10), (9, 10),
             (1, 11), (2, 11), (4, 11), (5, 11), (6, 11), (7, 11),
             ]


slow_movement = False
SIZE = 15
HM_EPISODES = 200000

MOVE_PENALTY = -1
CAN_NOT_MOVE_PENALTY = -300
# DISTANCE_REDUCE_REWARD = 1
# 其实这里的设计不太对，如果移除箱子，应该有惩罚。
DEST_CLOSE_REWARD = 1
DEST_1_REWARD = 3000

epsilon = 0.1

EPS_DECAY = 0.99  # Every episode will be epsilon*EPS_DECAY

SHOW_EVERY = 1000  # how often to play through env visually.

# start_q_table = "pushbox3-1590878109.pickle"  # None or Filename
start_q_table = None  # None or Filename


LEARNING_RATE = 0.2
DISCOUNT = 0.95

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
            return False
    return True


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
                    return True
        else:
            # 没有box
            res = has_wall_in_front(self.x, self.y, choice)
            if not res:
                # 人物往该方向移动一格
                move(player, None, choice)
                return True
        return False

    def position(self):
        return self.x, self.y


# 从all_state表里取出来赋值
if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    # 改为用时初始化

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

# print(q_table)


def is_box_in_dead_road(box_list, dest_list):
    # 查找box是否有两边都在state_list之外
    for box in box_list:
        x, y = box.position()
        if (x, y) in dead_state:
            return True

    # 是否堵在路口
    count = 0
    for box in box_list:
        x, y = box.position()
        if (x, y) in [(4, 4), (5, 4), (6, 4), (7, 4)]:
            count += 1
    if count > 2:
        return True

    return False


def take_action(player, action, box_list, dest_list, last_arrive_count, last_distance):
    reward = 0

    # 这里要把箱子传进去。因为玩家的移动可能会影响箱子的位置。
    res = player.action(action, box_list)

    # 撞墙惩罚
    if res == False:
        reward += CAN_NOT_MOVE_PENALTY

    # 箱子推入死角的惩罚
    is_dead = is_box_in_dead_road(box_list, dest_list)
    if is_dead == True:
        reward += CAN_NOT_MOVE_PENALTY

    # 判断3个箱子有多少个在dest中,然后计算总的reward
    count = 0
    for box in box_list:
        for dest in dest_list:
            if box.position() == dest.position():
                count += 1

    distance = 0
    # 计算总距离,变短则+1
    for box in box_list:
        x = box.x
        y = box.y
        dx = dest_list[0].x
        dy = dest_list[0].y
        distance += (x-dx)**2 + (y-dy)**2

    if distance - last_distance < 0:
        reward += 1
    else:
        reward -= 1
    # 计算到达光标点的数量是否变化,如果推出去了，是要扣分的
    diff = count - last_arrive_count
    reward += diff * DEST_1_REWARD
    if count == len(box_list):
        # 35 自主探索的概率要不断的降低。有意思。可能实际应用中都是这样吧。当一个人越来越有经验后，就越不需要自主探索。
        # 只有成功的结果才会让episilon下降
        global epsilon
        epsilon *= EPS_DECAY

    last_arrive_count = count
    last_distance = distance

    if count == len(box_list) or res == False or is_dead:

        return reward, last_arrive_count, last_distance, True
    else:
        return reward, last_arrive_count, last_distance, False


def get_qtable_value(state):
    if state not in q_table:
        q_table[state] = [np.random.uniform(-5, 0) for i in range(4)]

    return q_table[state]


def update_q_value(reward, player, box_list, old_state, action):
    pos_list = [player.position()]
    for box in box_list:
        pos_list.append(box.position())
    new_pos = tuple(pos_list)
    max_future_q = np.max(get_qtable_value(new_pos))

    # 26 把当前相对位置状态和action下对应的收益qvalue取了出来。
    # 这个要思考一下是不是new_pos。 不是是老的position位置
    # 这里错了。哎，应该是老位置，不是新位置.
    current_q = get_qtable_value(old_state)[action]

    new_q = (1 - LEARNING_RATE) * current_q + \
        LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    get_qtable_value(old_state)[action] = new_q
    return new_q


def show_movement(player, box_list, dest_list):
    # 前面二维是x，y，第三的维度是颜色的索引。其实直接传颜色进去也行吧。本来颜色。
    # 用这种方式定义了一个环境，x值，y值，以及格子颜色
    # starts an rbg of our size
    # 首先把所有区域设置成墙
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    # 然后修改可达点的颜色
    for x, y in all_state:
        env[y, x] = d[ROAD_N]

    # 然后再修改玩家、box、dest的颜色
    # sets the dest location to red
    for dest in dest_list:
        env[dest.y][dest.x] = d[DEST_N]

    # sets the player tile to blue
    env[player.y][player.x] = d[PLAYER_N]

    # sets the box location tile to green color
    for box in box_list:
        env[box.y][box.x] = d[BOX_N]

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
    # if reward == DEST_3_REWARD:
    #     # 32 ord是什么意思？ 点q键。返回q的unicode字符编码。
    #     if cv2.waitKey(500) & 0xFF == ord('q'):
    #         return False
    # else:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    if slow_movement:
        sleep(0.1)


# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example


episode_rewards = []


def get_selectable_random_action(player, box_list):
    c_list = []
    for choice in range(4):
        box = has_box_in_front(player.x, player.y, choice, box_list)
        if box:
            # 判断box的前面还有没有其他box或者墙
            res = has_box_in_front(box.x, box.y, choice, box_list)
            if not res:
                res = has_wall_in_front(box.x, box.y, choice)
                if not res:
                    # 可以移动
                    c_list.append(choice)
        else:
            # 没有box
            res = has_wall_in_front(player.x, player.y, choice)
            if not res:
                # 可以移动
                c_list.append(choice)

    # print(c_list)
    action = random.choice(c_list)
    return action


for episode in range(HM_EPISODES):
    player = Blob(11, 6)

    # TODO 测试一个
    box_list = [Blob(8, 4), Blob(9, 4), Blob(10, 5), Blob(11, 6)]
    dest_list = [Blob(1, 4), Blob(2, 4), Blob(1, 5), Blob(2, 5)]
    # dest_list = [Blob(1, 4), Blob(2, 4), Blob(1, 5), Blob(2, 5), Blob(1, 6), Blob(
    #     2, 6), Blob(1, 7), Blob(2, 7), Blob(1, 8), Blob(2, 8), Blob(2, 9)]
    if episode % SHOW_EVERY == 0 and episode != 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(
            f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    last_distance = 0
    last_arrive_count = 0
    for i in range(300):
        pos_list = [player.position()]
        for box in box_list:
            pos_list.append(box.position())
        old_state = tuple(pos_list)
        # 通过当前的位置计算出最大q-value的行动
        if np.random.random() > epsilon:
            action = np.argmax(get_qtable_value(old_state))
        else:
            action = get_selectable_random_action(player, box_list)
            # action = np.random.randint(0, 4)
        # Take the action!
        reward, last_arrive_count, last_distance, finish = take_action(player, action, box_list,
                                                                       dest_list, last_arrive_count, last_distance)

        # NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        update_q_value(reward, player, box_list, old_state, action)

        if show:
            show_movement(player, box_list, dest_list)

        # 用于计算平均reward
        episode_reward += reward

        # 33 结束.碰到敌人或者碰到食物。 推箱子中也是，箱子碰到目标点就结束。
        if finish:
            break

    # print(episode_reward)
    # 34 rewards会呈现怎样的规律？越来越高
    episode_rewards.append(episode_reward)


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
with open(f"pushbox3-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
