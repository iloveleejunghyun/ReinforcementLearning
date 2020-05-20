'''
process
1、首先初始化数据。RL学习的几个参数。然后是q-table. 这里比较特殊，以往是state-action的二维数组, 但是这个是个dict只有state，值为数据。
2、q-table也可以从存储数据中取出来。接着用。哈哈
3、然后就是嵌套循环，外层每次为一集episode。内层每次为1步，移动一个格子。
    3.1、英雄每次的action由探索率决定，如果决定探索，则随机走。
    如果决定不探索，则取出该state下对应的最大价值action(从q-table值的list里面取)。
    3.2、走了这个action后，计算得到的回报reward，加上原本的回报q-value,加上对未来的预期reward，计算出一个新的q-value,更新老的state和action(后者没变其实)下的q-value值。这样就达到了每次增加一点点或者减少一点点q-value。
4、这样一直走，该显示的时候就画图。
5、最后存储q-table就行了。

'''


import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

# 1下面这句话的意思？ 使用一种画图风格 ggplot,
# 注释掉和启动没什么区别呀。
# style.use("ggplot")

# 2 指的是q-table的长宽(是个正方形), 游戏棋盘的大小
SIZE = 10

HM_EPISODES = 25000

# 3 能不能把以下两项变成负数？ 能
MOVE_PENALTY = -1
ENEMY_PENALTY = -300
FOOD_REWARD = 25

# 4 这个是探索率？0.9挺高的
epsilon = 0.9

# 5 探索率会逐步降低。在开始学习的时候，因为代理不清楚套路，所以容易走进局部最优解。用高探索率可以避免这个问题。
# 但是等代理逐步成熟后，就不需要那么多探索了。 多了反而是浪费时间。
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY

SHOW_EVERY = 3000  # how often to play through env visually.

# start_q_table = "qtable-1588121034.pickle" # None or Filename
start_q_table = None  # None or Filename

# 6学习率，有多少比重部分使用新的reward
LEARNING_RATE = 0.1
# 7 将来预测reward的折扣。 这是一个未来期待值。
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

# 8 小方块。用于定义玩家、食物和敌人


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    # 9 定义打印本类的实例，print
    def __str__(self):
        return f"{self.x}, {self.y}"

    # 10 定义运算符-
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    # 11 行动，上下左右。
    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

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

        # 12 越界行动是无效的。 reward会有什么反应？ 要去看外面. 反应和上次的相同。这个位置该减多少reward，就-多少。
        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


# ?13 初始化 q-table. 里面的坐标的变化范围是-9~10?怎么可以为负数？
# 14 最后一句看不懂.看下面
# 20 的4次方。有点大。有这么多状态吗？有
# 状态和action都是从-9到10的元组.错误。这个q-table构造的不是state-action表了
# q-table的action不是只有4种吗？怎么这里有20*20种？ 看下面。
# 好像理解错了。这个q-table是个dict。里面只有一对一的映射。前面一对元组是玩家距离食物的距离(横纵坐标)， 后面元组是玩家到敌人的横纵坐标。 窝草把所有的情况都列出来了。16万种情况。不知道王者是不是这么做的？
# 搞错了，里面的值对应4个action的q-value
# ？ 有没有简单一点的办法？直接算距离应该更简单点，这个可以给自己留一道题来做哈哈。
# 生成的键值对是酱紫的:((9, 9), (3, 9)): [-0.9395273916153881, -1.8960367424225528, -0.6528177115349028, -1.7907413080366292]
# key是一个元组，这个元组里面包含了两个元组。第一个小元组是到食物的距离？ 第二个小元组是到敌人的距离？这种用距离来计算的好处是，即使是食物和敌人改变了位置，也能相对的找到食物避免敌人。
if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                for iiii in range(-SIZE+1, SIZE):
                    q_table[((i, ii), (iii, iiii))] = [
                        np.random.uniform(-5, 0) for i in range(4)]  # 里面初值全是负数，怎么都都是负反馈。 这里面存的是4个值的数组。代表的是当前状态下，走上下左右四种情况的q-value值。哎好复杂。和之前的不太一样。


# 15 为什么这个表这么大？28*20*20*20，你说大不大。16万
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

# print(q_table)

# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

# 16 把reward收集起来的目的是观察规律。是的，要的结果是，随着训练，代理越来越熟练，每集结束时平均反馈越来越高。
episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        # 17 这里统计了平均reward.为什么会下降？ 不是下降，而是上升。因为越来越熟练了，错误越来越少。
        print(
            f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        # 18 计算玩家和食物以及敌人的距离
        obs = (player-food, player-enemy)
        # print(obs)
        # 19 epsilon指的是探索概率。
        if np.random.random() > epsilon:
            # GET THE ACTION
            # 20 q-table中的纵坐标state存的是玩家相对食物的距离以及玩家相对敌人的距离的元组tuple。
            # 在推箱子中可以做成玩家相对箱子的距离以及箱子相对目标点的距离。
            # 这里找出了那行中最大值的action
            # print(q_table[obs], type(q_table[obs]))
            action = np.argmax(q_table[obs])
        else:
            # 21 action只有4种0,1,2,3
            action = np.random.randint(0, 4)
        # Take the action!
        player.action(action)

        #### MAYBE ###
        # enemy.move()
        # food.move()
        ##############

        # 这个代码封装一下会不会好一点，计算reward也。
        if player.x == enemy.x and player.y == enemy.y:
            # 22 这个地方可以直接用负数
            reward = ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            # 23 给的当前reward只有3种状态。要么与敌人重合，惩罚；要么与食物重合，奖励；要么都不重合，惩罚1
            # 推箱子中。可以做成，箱子与目标点重合，最大奖励；箱子没与目标点重合，小惩罚1；人物被锁死(暂时做不出来)
            reward = MOVE_PENALTY
        # NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        # 24 顺序：先判断相对位置，然后走最大收益行动，然后又判断相对位置，又找最大收益行动？
        new_obs = (player-food, player-enemy)
        # 25 这个取出来最大收益，而不是最大收益对应的行动。是的用于计算新的收益。
        max_future_q = np.max(q_table[new_obs])
        # 26 把当前相对位置状态和action下对应的收益qvalue取了出来。
        current_q = q_table[obs][action]

        # 27 新q-value是固定的几种情况。如果找到食物，那就固定为最大收益。
        # 不然的话，就用公式计算新收益。这里就是限制了最大收益。
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + \
                LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            # 28 传入了3元组？，输出是个3维数组。前面二维是x，y，第三的维度？是颜色的索引。其实直接传颜色进去也行吧。本来颜色。
            # 用这种方式定义了一个环境，x值，y值，以及格子颜色
            # starts an rbg of our size
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            # 29 设置颜色。数组中的每个位置的值是颜色。
            # sets the food location tile to green color
            env[food.x][food.y] = d[FOOD_N]
            # sets the player tile to blue
            env[player.x][player.y] = d[PLAYER_N]
            # sets the enemy location to red
            env[enemy.x][enemy.y] = d[ENEMY_N]
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
            if reward == FOOD_REWARD or reward == ENEMY_PENALTY:
                # 32 ord是什么意思？ 点q键。返回q的unicode字符编码。
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 32 episode_reward是指每一集的总reward, 意义何在？ 后面要计算平均值，用于统计是否进步。
        episode_reward += reward

        # 33 结束.碰到敌人或者碰到食物。 推箱子中也是，箱子碰到目标点就结束。
        if reward == FOOD_REWARD or reward == ENEMY_PENALTY:
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
