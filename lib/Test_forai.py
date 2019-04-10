import pygame
from pygame.locals import *
import sys
import random
import time
import numpy as np
class Game_Config():
    MOVE_STAY = np.array([1, 0, 0,0,0])
    MOVE_LEFT = np.array([0, 1, 0,0,0])
    MOVE_RIGHT = np.array([0, 0, 1,0,0])
    MOVE_UP = np.array([0,0,0,1,0])
    MOVE_DOWN = np.array([0,0,0,0,1])
    # 只有5種type的ACTION
    ACTION_DIM = 5
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255,0,0)
    SCREEN_SIZE = [200,300]
    BALL_SIZE = [15, 15]
    BULLTE_SIZE = [3,15]
    PLAYER_SIZE = [10,10]
    SCREEN_SIZE_MID = SCREEN_SIZE[0] / 2
    PLAYERCOLOR =(222, 222, 222)
    YELLOW = (255,255,0)
    RECOVERING_TIME = 1000
    RECOVERED_TIME = 0
    MOVE_X_WAY = [3,-3 , 0]
    MOVE_Y_WAY = [3,-3]
pass

class Balls():
    def __init__(self, i_pos_x, i_pos_y, i_dir_x, i_dir_y , i_hp , i_size_x ,i_size_y ):
        self.pos_x = i_pos_x
        self.pos_y = i_pos_y
        self.dir_x = i_dir_x
        self.dir_y = i_dir_y
        self.rect = pygame.Rect(self.pos_x, self.pos_y, i_size_x, i_size_y)
        self.Isdead = False
        self.hp = i_hp
    pass
    def update(self):
        self.rect.bottom += self.dir_y
        self.rect.left+= self.dir_x
        if (self.rect.top <= 0):
            self.dir_y *= -1
        if (self.rect.left <= 0 or self.rect.right >= Game_Config.SCREEN_SIZE[0]):
            self.dir_x *= -1
        if (self.rect.bottom >= Game_Config.SCREEN_SIZE[1]):
            self.Isdead = True
    pass

    def hitted(self , damage):
        self.hp -=damage
        if (self.hp <=0):
            self.Isdead = True
    pass
pass

class Enemy(Balls):
    def __init__(self , i_pos_x, i_pos_y, i_dir_x, i_dir_y , i_hp ):
        super().__init__(i_pos_x , i_pos_y , i_dir_x , i_dir_y , i_hp , Game_Config.BALL_SIZE[0] , Game_Config.BALL_SIZE[1])
    pass
pass

class Bullets():
    def __init__(self , i_pos_x , i_pos_y):
        super().__init__()
        self.pos_x = i_pos_x
        self.pos_y = i_pos_y
        self.dir_y = -5
        self.rect = pygame.Rect(self.pos_x , self.pos_y ,Game_Config.BULLTE_SIZE[0] , Game_Config.BULLTE_SIZE[1] )
    pass
    def update(self):
        self.rect.bottom += self.dir_y
    pass
pass

class Player(Balls):
    def __init__(self, i_pos_x, i_pos_y):
        super().__init__(i_pos_x , i_pos_y , 0,0,3,Game_Config.PLAYER_SIZE[0] , Game_Config.PLAYER_SIZE[1])
        self.bullet_list = []
        self.recovering = False
    pass

    def shot(self):
        newbullet = Bullets(self.rect.left+5, self.rect.top)
        self.bullet_list.append(newbullet)
    pass

    def hitted(self):
        self.hp -=1
        if (self.hp <=0):
            self.Isdead = True
    pass

    def check_Edge(self):
        self.IsEdge =False
        if (self.rect.top <= 0):
            self.rect.top = 0
        if (self.rect.bottom >= Game_Config.SCREEN_SIZE[1]):
            self.rect.bottom = Game_Config.SCREEN_SIZE[1]
        if (self.rect.left <=0):
            self.rect.left = 0
        if (self.rect.right >= Game_Config.SCREEN_SIZE[0]):
            self.rect.right = Game_Config.SCREEN_SIZE[0]
        pass
    pass


    def move_left(self):
        self.rect.left = self.rect.left - 5
        pass
    pass

    def move_right(self):
        self.rect.left = self.rect.left + 5
        pass
    pass

    def move_up(self):
        self.rect.top = self.rect.top - 5
        pass
    pass

    def move_down(self):
        self.rect.bottom = self.rect.bottom + 5
        pass
    pass
pass

class Game():
    ball_list = []
    bullet_list = []
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(Game_Config.SCREEN_SIZE)
        pygame.display.set_caption('Simple Game')
        self.ball_list.clear()
        self.bullet_list.clear()
        for i in range(5):
            self.Add_Enemy(random.randint(Game_Config.BALL_SIZE[0],
                                         Game_Config.SCREEN_SIZE[0] - Game_Config.BALL_SIZE[0] -15),
                           random.randint(Game_Config.BALL_SIZE[1],
                                         int(Game_Config.SCREEN_SIZE[1] / 3) + Game_Config.BALL_SIZE[1]))
        pass
        self.player = Player(Game_Config.SCREEN_SIZE_MID + 30, Game_Config.SCREEN_SIZE[1] - 30)
        self.score = 0
        pygame.time.set_timer(USEREVENT + 1, 500)
        pygame.time.set_timer(USEREVENT + 2, 870)
    pass

    def Add_Enemy(self , i_pos_x ,i_pos_y):
        self.ball_pos_x = i_pos_x
        self.ball_pos_y = i_pos_y
        self.ball_dir_x = Game_Config.MOVE_X_WAY[random.randint(0,2)]  # -1=left 1=right
        self.ball_dir_y = -2  # -1=up 1=down
        self.ball = Enemy(self.ball_pos_x, self.ball_pos_y, self.ball_dir_x, self.ball_dir_y, 2)
        self.ball_list.append(self.ball)
    pass

    def draw_text(self , screen , text , x , y , color = (255,255,255)):
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render(text, False, color)
        screen.blit(textsurface, (x, y))
    pass

    def detect_collision_player(self):
        for enemy in self.ball_list:
            if (pygame.Rect.colliderect(self.player.rect, enemy.rect)):
                if (self.player.recovering == False):
                    self.player.recovering = True
                    self.player.hitted()
                    Game_Config.RECOVERED_TIME = time.time()
                    return True
                else:
                    return False
            pass
        return False
    pass

    def detect_collision_enemy(self):
        for bullet in self.player.bullet_list:
            for enemy in self.ball_list:
                if (pygame.Rect.colliderect(bullet.rect, enemy.rect) and self.player.bullet_list.__contains__(bullet)):
                    self.player.bullet_list.remove(bullet)
                    enemy.hitted(1)
                    self.score += 1
                    return True
                pass
            pass
        return False
        pass
    pass

    def Check_enemy_dead(self):
        for enemy in self.ball_list:
            if (enemy.Isdead):
                self.ball_list.remove(enemy)
                self.score += 5
        pass
    pass

    def current_game_img(self):
        return pygame.surfarray.array3d(pygame.display.get_surface())
    pass

    def step(self, action, text="test"):
        for event in pygame.event.get():
            #if event.type == QUIT:
                #pygame.quit()
                #sys.exit()
            if self.player.Isdead:
                Game.__init__(self)
            if event.type == USEREVENT+1:
                self.player.shot()
            if event.type == USEREVENT+2:
                if (len(self.ball_list) <5):
                    self.Add_Enemy(random.randint(Game_Config.BALL_SIZE[0],
                                         Game_Config.SCREEN_SIZE[0] - Game_Config.BALL_SIZE[0] -15), 15)
        # action是MOVE_STAY、MOVE_LEFT、MOVE_RIGHT、MOVE_UP、MOVE_DOWN
        # ai控制腳色上下左右；返回游戏界面像素数和对应的奖励。(像素->奖励->强化角色往奖励高的方向移动)
        if np.array_equal(action, Game_Config.MOVE_LEFT):
            self.player.move_left()
        elif np.array_equal(action, Game_Config.MOVE_RIGHT):
            self.player.move_right()
        elif np.array_equal(action , Game_Config.MOVE_UP):
            self.player.move_up()
        elif np.array_equal(action , Game_Config.MOVE_DOWN):
            self.player.move_down()
        else:
            pass
        pass
        self.player.check_Edge()
        self.Check_enemy_dead()

        terminal =  False
        if (self.player.Isdead):
            terminal = True
        if self.player.recovering:
            Game_Config.PLAYERCOLOR = Game_Config.RED
        else:
            Game_Config.PLAYERCOLOR = Game_Config.YELLOW
        pass
        time_pass = time.time() - Game_Config.RECOVERED_TIME
        if self.player.recovering and (time_pass) >= 3:
            self.player.recovering = False
        pass
        self.screen.fill(Game_Config.BLACK)
        pygame.draw.rect(self.screen, Game_Config.PLAYERCOLOR, self.player.rect)
        for i in range(len(self.ball_list)):
            self.ball_list[i].update()
            pygame.draw.rect(self.screen, Game_Config.WHITE, self.ball_list[i].rect)
            pass
        pass

        for bullets in self.player.bullet_list:
            bullets.update()
            pygame.draw.rect(self.screen, Game_Config.RED, bullets.rect)
        pass

        reward = 0
        IsCollide  = self.detect_collision_player()

        if (self.detect_collision_enemy() and not(IsCollide)):
            reward += 5  # 击中奖励
        if (IsCollide):
            #player_pos = abs(self.player.rect.right + self.player.rect.left) / 2.0
            #enemy_pos = abs(CollideEnemy.rect.right + CollideEnemy.rect.left) / 2.0
            # reward = -5*((abs( bar_pos -  ball_pos ))/(Game_Config.BAR_SIZE))
            #reward = -10 * (Game_Config.BALL_SIZE[0] / abs(player_pos - enemy_pos))
            reward += -20
            # reward = -1
        #elif(not(IsCollide)):
            #reward = 1
        pass
        #print(reward)
        # show msg (如果有分數的話，要連分數一起抓，必須放在面相素前面，這裡不考慮分數，所以放後面)
        # self.show_text(text)
        # 获得游戏界面像素
        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        self.clock.tick(60)
        # 返回游戏界面像素和对应的奖励
        return screen_image, reward, terminal
    pass

    def run (self):
        pygame.mouse.set_visible(0)
        pygame.time.set_timer(USEREVENT+1 , 500)
        player_move_left = False
        player_move_right = False
        player_move_up = False
        player_move_down = False
        while (True):

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == USEREVENT+1:
                    self.player.shot()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:  # 鼠标左键按下(左移)
                    player_move_left = True
                elif event.type == pygame.KEYUP and event.key == pygame.K_LEFT:  # 鼠标左键释放
                    player_move_left = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:  # 右键
                    player_move_right = True
                elif event.type == pygame.KEYUP and event.key == pygame.K_RIGHT:
                    player_move_right = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:  # 鼠标上键按下(左移)
                    player_move_up = True
                elif event.type == pygame.KEYUP and event.key == pygame.K_UP:  # 鼠标上键释放
                    player_move_up = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:  # 下键
                    player_move_down = True
                elif event.type == pygame.KEYUP and event.key == pygame.K_DOWN:
                    player_move_down = False
                pass
            if player_move_left == True and player_move_right == False:
                self.player.move_left()
            if player_move_left == False and player_move_right == True:
                self.player.move_right()
            if player_move_up == True and player_move_down == False:
                self.player.move_up()
            if player_move_up == False and player_move_down == True:
                self.player.move_down()
            self.player.check_Edge()
            if self.player.recovering:
                Game_Config.PLAYERCOLOR = Game_Config.RED
            else:
                Game_Config.PLAYERCOLOR = (222,222,222)
            pass
            time_pass = time.time() - Game_Config.RECOVERED_TIME
            print(time_pass)
            if self.player.recovering and (time_pass) >= 3:
                self.player.recovering = False
            pass

            self.screen.fill(Game_Config.BLACK)
            if (not(self.player.Isdead)):
                self.draw_text(self.screen ,'HP:' + str(self.player.hp) , 10 ,10)
                print(str(self.player.hp))
            else:
                self.draw_text(self.screen , 'Game over:' + str(self.score) , Game_Config.SCREEN_SIZE[0]/2 -90 , Game_Config.SCREEN_SIZE[1] /2)
                print('Game over : ' + str(self.score))
                pygame.display.update()
                self.clock.tick(60)
                continue
            pygame.draw.rect(self.screen, Game_Config.PLAYERCOLOR, self.player.rect)
            for i in range(len(self.ball_list)):
                self.ball_list[i].update()
                pygame.draw.rect(self.screen ,Game_Config.WHITE , self.ball_list[i].rect)
                pass
            pass
            for bullets in self.player.bullet_list:
                bullets.update()
                pygame.draw.rect(self.screen, Game_Config.RED, bullets.rect)
            pass
            self.detect_collision_player()
            IsCollide  , CollidedEnemy =self.detect_collision_enemy()
            pygame.display.update()
            self.clock.tick(60)
    pass
