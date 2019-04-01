import pygame
from pygame.locals import *
import sys
import random
import time
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255,0,0)
SCREEN_SIZE = [320, 400]
BAR_SIZE = [20, 5]
BALL_SIZE = [15, 15]
BULLTE_SIZE = [3,15]
PLAYER_SIZE = [30,30]

SCREEN_SIZE_MID = SCREEN_SIZE[0] / 2
PLAYERCOLOR =(222, 222, 222)
PLAYERCOLOR2 =(255, 68, 0)
RECOVERING_TIME = 1000
RECOVERED_TIME = 0
class Balls():
    def __init__(self, i_pos_x, i_pos_y, i_dir_x, i_dir_y):
        super().__init__()
        self.pos_x = i_pos_x
        self.pos_y = i_pos_y
        self.dir_x = i_dir_x
        self.dir_y = i_dir_y
        self.rect = pygame.Rect(self.pos_x, self.pos_y, BALL_SIZE[0], BALL_SIZE[1])
    pass
    def update(self):
        self.rect.bottom += self.dir_y
        self.rect.left+= self.dir_x
        if (self.rect.top <= 0 or self.rect.bottom >= SCREEN_SIZE[1]):
            self.dir_y *= -1
        elif (self.rect.left <= 0 or self.rect.right >= SCREEN_SIZE[0]):
            self.dir_x *= -1
    pass
pass

class Bullets():
    def __init__(self , i_pos_x , i_pos_y):
        super().__init__()
        self.pos_x = i_pos_x
        self.pos_y = i_pos_y
        self.dir_y = -1
        self.rect = pygame.Rect(self.pos_x , self.pos_y ,BULLTE_SIZE[0] , BULLTE_SIZE[1] )
    pass
    def update(self):
        self.rect.bottom += self.dir_y
    pass
pass

class Player:
    def __init__(self, i_pos_x, i_pos_y):
        super().__init__()
        self.pos_x = i_pos_x
        self.pos_y = i_pos_y
        self.rect = pygame.Rect(self.pos_x, self.pos_y, PLAYER_SIZE[0], PLAYER_SIZE[1])
        self.bullet_list = []
        self.recovering = False
    pass
    def shot(self):
        newbullet = Bullets(self.rect.left+15, self.rect.top)
        self.bullet_list.append(newbullet)
    pass
pass
class Game():
    ball_list = []
    bullet_list = []
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption('Simple Game')
        for i in range(10):
            self.ball_pos_x = random.randint(BALL_SIZE[0],SCREEN_SIZE[0]- BALL_SIZE[0])
            self.ball_pos_y = random.randint(BALL_SIZE[1],SCREEN_SIZE[1]- BALL_SIZE[1])
            self.ball_dir_x = -2 #-1=left 1=right
            self.ball_dir_y = -3 # -1=up 1=down
            self.ball = Balls(self.ball_pos_x,self.ball_pos_y,self.ball_dir_x,self.ball_dir_y)
            self.ball_list.append(self.ball)
        pass
    pass
    def player_move(self , way , I_value):
        ways = [self.player.rect.left , self.player.rect.bottom,self.player.rect.top , self.player.rect.bottom]
        ways[way] += I_value
    pass

    def player_move_left(self):
        self.player.rect.left = self.player.rect.left - 5
    pass

    def player_move_right(self):
        self.player.rect.right = self.player.rect.right + 5
    pass

    def player_move_up(self):
        self.player.rect.top = self.player.rect.top - 5
    pass

    def player_move_down(self):
        self.player.rect.bottom = self.player.rect.bottom + 5
    pass
    def run (self):
        self.player = Player(SCREEN_SIZE_MID + 30, SCREEN_SIZE[1] - 30)
        def addbulllet(player):
            self.player.shot()
        pass
        def detect_conlision():
            for bullet in self.player.bullet_list:
                for enemy in self.ball_list:
                    if (pygame.Rect.colliderect(bullet.rect , enemy.rect) and self.player.bullet_list.__contains__(bullet)):
                        self.player.bullet_list.remove(bullet)
                    pass
                pass
            pass
            for enemy in self.ball_list:
                if (pygame.Rect.colliderect(self.player.rect , enemy.rect) and (self.player.recovering!=True)):
                    print("good")
                    self.player.recovering = True
                    global RECOVERED_TIME
                    RECOVERED_TIME =time.time()
                pass
        pass
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
                    addbulllet(self.player)
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
                self.player_move_left()
            if player_move_left == False and player_move_right == True:
                self.player_move_right()
            if player_move_up == True and player_move_down == False:
                self.player_move_up()
            if player_move_up == False and player_move_down == True:
                self.player_move_down()
            if self.player.recovering:
                global PLAYERCOLOR
                PLAYERCOLOR = RED
            else:
                PLAYERCOLOR = (222,222,222)
            pass
            time_pass = time.time() - RECOVERED_TIME
            print(time_pass)
            if self.player.recovering and (time_pass) >= 3:
                self.player.recovering = False
            pass
            self.screen.fill(BLACK)
            detect_conlision()
            print(PLAYERCOLOR)
            pygame.draw.rect(self.screen, PLAYERCOLOR, self.player.rect)

            for i in range(len(self.ball_list)):
                self.ball_list[i].update()
                pygame.draw.rect(self.screen ,WHITE , self.ball_list[i].rect)
                pass
            pass
            for bullets in self.player.bullet_list:
                bullets.update()
                pygame.draw.rect(self.screen, RED, bullets.rect)
            pass

            pygame.display.update()
            self.clock.tick(60)
    pass
game = Game()
game.run()