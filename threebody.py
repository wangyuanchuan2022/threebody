import json
import math
import random
import sys
import time as t

import cv2
import pygame
import numpy as np
from scipy.integrate import solve_ivp


class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, Vector2D):
            return self.x * other.x + self.y * other.y
        else:
            return Vector2D(self.x * other, self.y * other)

    def __truediv__(self, other):
        return Vector2D(self.x / other, self.y / other)

    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __str__(self):
        return f"({self.x:.2f}, {self.y:.2f})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise IndexError("Vector2D has only two dimensions")

    def __setitem__(self, item, value):
        if item == 0:
            self.x = value
        elif item == 1:
            self.y = value
        else:
            raise IndexError("Vector2D has only two dimensions")


class Sun:
    def __init__(self, position: tuple[float, float], velocity: tuple[float, float], mass: float, temperature: float,
                 i):
        self.pos = Vector2D(position[0], position[1])
        self.v = Vector2D(velocity[0], velocity[1])
        self.m = mass
        self.temp = temperature
        self.r = math.sqrt(mass) * 0.9
        self.i = i
        img = np.zeros([100, 100, 4])
        for i in range(100):
            for j in range(100):
                try:
                    d = ((i - 50) ** 2 + (j - 50) ** 2) ** 0.5 - self.r
                    num = int(250 * 0.8 ** d)
                    img[i, j, 0] = num
                    img[i, j, 1] = num
                    img[i, j, 2] = num
                    img[i, j, 3] = num
                except:
                    pass
        cv2.imwrite(f'sun.png', img)
        self.img = pygame.image.load(f'sun.png')


def distance(a: Vector2D, b: Vector2D):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def d_func(t, y, m, G=700):
    if len(y) == 16:
        x1, y1, v1x, v1y, x2, y2, v2x, v2y, x3, y3, v3x, v3y, x4, y4, v4x, v4y = y
        m1, m2, m3, m4 = m
        a12x = (x2 - x1) / ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 1.5 * G
        a13x = (x3 - x1) / ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 1.5 * G
        a14x = (x4 - x1) / ((x1 - x4) ** 2 + (y1 - y4) ** 2) ** 1.5 * G
        a23x = (x3 - x2) / ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 1.5 * G
        a24x = (x4 - x2) / ((x4 - x2) ** 2 + (y4 - y2) ** 2) ** 1.5 * G
        a34x = (x4 - x3) / ((x4 - x3) ** 2 + (y4 - y3) ** 2) ** 1.5 * G

        a12y = (y2 - y1) / ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 1.5 * G
        a13y = (y3 - y1) / ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 1.5 * G
        a14y = (y4 - y1) / ((x1 - x4) ** 2 + (y1 - y4) ** 2) ** 1.5 * G
        a23y = (y3 - y2) / ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 1.5 * G
        a24y = (y4 - y2) / ((x4 - x2) ** 2 + (y4 - y2) ** 2) ** 1.5 * G
        a34y = (y4 - y3) / ((x4 - x3) ** 2 + (y4 - y3) ** 2) ** 1.5 * G

        dv1x = a12x * m2 + a13x * m3 + a14x * m4
        dv2x = -a12x * m1 + a23x * m3 + a24x * m4
        dv3x = -a13x * m1 - a23x * m2 + a34x * m4
        dv4x = -a14x * m1 - a24x * m2 - a34x * m3

        dv1y = a12y * m2 + a13y * m3 + a14y * m4
        dv2y = -a12y * m1 + a23y * m3 + a24y * m4
        dv3y = -a13y * m1 - a23y * m2 + a34y * m4
        dv4y = -a14y * m1 - a24y * m2 - a34y * m3

        return [v1x, v1y, dv1x, dv1y,
                v2x, v2y, dv2x, dv2y,
                v3x, v3y, dv3x, dv3y,
                v4x, v4y, dv4x, dv4y]

    elif len(y) == 12:
        x1, y1, v1x, v1y, x2, y2, v2x, v2y, x3, y3, v3x, v3y = y
        m1, m2, m3 = m
        a12x = (x2 - x1) / ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 1.5 * G
        a13x = (x3 - x1) / ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 1.5 * G
        a23x = (x3 - x2) / ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 1.5 * G

        a12y = (y2 - y1) / ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 1.5 * G
        a13y = (y3 - y1) / ((x1 - x3) ** 2 + (y1 - y3) ** 2) ** 1.5 * G
        a23y = (y3 - y2) / ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 1.5 * G

        dv1x = a12x * m2 + a13x * m3
        dv2x = -a12x * m1 + a23x * m3
        dv3x = -a13x * m1 - a23x * m2

        dv1y = a12y * m2 + a13y * m3
        dv2y = -a12y * m1 + a23y * m3
        dv3y = -a13y * m1 - a23y * m2

        return [v1x, v1y, dv1x, dv1y,
                v2x, v2y, dv2x, dv2y,
                v3x, v3y, dv3x, dv3y]

    if len(y) == 8:
        x1, y1, v1x, v1y, x2, y2, v2x, v2y = y
        m1, m2 = m
        a12x = (x2 - x1) / ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 1.5 * G

        a12y = (y2 - y1) / ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 1.5 * G

        dv1x = a12x * m2
        dv2x = -a12x * m1

        dv1y = a12y * m2
        dv2y = -a12y * m1

        return [v1x, v1y, dv1x, dv1y,
                v2x, v2y, dv2x, dv2y]

    if len(y) == 4:
        x1, y1, v1x, v1y = y

        return [v1x, v1y, 0, 0]


def update_p_temperature(p: Sun, sun_list: list[Sun], dt: float):
    k1 = 5e-10
    k2 = 2
    eps = 0.3
    v = 0
    for sun in sun_list:
        d = distance(sun.pos, p.pos)
        v += eps * (k1 * sun.temp ** 4 / d ** 2)
    v -= k2 * (p.temp - 3) * eps
    p.temp += v * dt


def num2angle(sinx, cosx) -> float:
    if cosx >= 0:
        return math.asin(sinx)
    elif sinx >= 0:
        return math.acos(cosx)
    else:
        return -math.acos(cosx)


def arrow(screen, lcolor, tricolor, start, end, trirad, thickness=2, rad=math.pi / 180):
    pygame.draw.line(screen, lcolor, start, end, thickness)
    rotation = (math.atan2(start[1] - end[1], end[0] - start[0])) + math.pi / 2
    pygame.draw.polygon(screen, tricolor, ((end[0] + trirad * math.sin(rotation),
                                            end[1] + trirad * math.cos(rotation)),
                                           (end[0] + trirad * math.sin(rotation - 120 * rad),
                                            end[1] + trirad * math.cos(rotation - 120 * rad)),
                                           (end[0] + trirad * math.sin(rotation + 120 * rad),
                                            end[1] + trirad * math.cos(rotation + 120 * rad))))


class Mainboard:
    def __init__(self):
        self.star_list = []
        with open('cfg.json', encoding='utf-8') as f:
            cfg = json.load(f)
        self.dt = cfg['dt']
        self.width = cfg['window_size'][0]
        self.height = cfg['window_size'][1]
        self.view_window = cfg['view_window']  # x y w h
        self.turn = 0
        self.time = 0
        self.cold = 0
        self.seed = 0
        self.choosed = 0
        self.show = True
        self.show_trace = True
        self.show_vw = True
        self.gm = False
        self._seed = cfg['seed'] if cfg['seed'] else 0
        self.start_speed = cfg['start_speed']
        self.per_frame = cfg['cal_per_frame']
        self.sun_list = []
        self.bg_l = []
        self.star_map = []
        self.state_l = []
        self.event_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.gm_code = [1073741906, 1073741905, 1073741906, 1073741905, 1073741904, 1073741903, 1073741904, 1073741903, 97, 98]
        self.trace = [[], [], [], []]
        self.colors = cfg['colors']

        pygame.init()
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Three Body')
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.f = pygame.font.Font('C:/Windows/Fonts/simhei.ttf', 30)
        self.f_s = pygame.font.Font('C:/Windows/Fonts/simhei.ttf', 15)

        self.reset()

    def reset(self):
        if self.time >= 100 or (self.time >= 50 and len(self.sun_list) == 3):
            with open('seed.txt', 'a') as f:
                f.write(f'seed: {self.seed} time: {self.time}\n')
        self.seed = random.randint(1, int(1e9)) if self._seed == 0 else self._seed
        # self.seed = 258817548  # 太阳系模拟器
        random.seed(self.seed)
        self.sun_list = []
        self.bg_l = []
        self.star_map = []
        self.state_l = []
        self.trace = [[], [], [], []]

        for i in range(3):
            sun = Sun((random.randint(300, 600), random.randint(300, 600)),
                      (-self.start_speed + random.random() * self.start_speed * 2,
                       -self.start_speed + random.random() * self.start_speed * 2), random.randint(90, 170),
                      6000, i=i)
            self.sun_list.append(sun)
            self.state_l.append(sun.pos.x)
            self.state_l.append(sun.pos.y)
            self.state_l.append(sun.v.x)
            self.state_l.append(sun.v.y)

        self.p = Sun((random.randint(300, 600), random.randint(300, 600)), (0.0, 20.0),
                     0.01, 273, -1)
        self.state_l.append(self.p.pos.x)
        self.state_l.append(self.p.pos.y)
        self.state_l.append(self.p.v.x)
        self.state_l.append(self.p.v.y)

        self.turn += 1
        self.time = 0
        self.cold = 0
        self.text = self.f.render(f"第{self.turn}轮文明 持续时间：{self.time} 种子:{self.seed}", True, (255, 255, 255),
                                  (0, 0, 0))
        for i in range(300):
            theta = -math.pi + random.random() * math.pi * 2
            y = random.random() * self.view_window[3]
            ld = random.randint(50, 255)
            self.star_map.append([theta, y, ld])
            self.star_map.append([-math.pi * 2 + theta, y, ld])
            self.star_map.append([math.pi * 2 + theta, y, ld])

        self.fusion()

    def update(self):
        # update the position of the suns
        res = solve_ivp(d_func, (0, self.dt * 10), self.state_l, t_eval=[self.dt * 10],
                        args=([s.m for s in self.sun_list] + [self.p.m],), first_step=self.dt)
        y = np.array(res.y)
        y = y.reshape([len(self.state_l)])
        self.state_l = y
        for i, sun in enumerate(self.sun_list):
            sun.pos.x = y[4 * i]
            sun.pos.y = y[4 * i + 1]
            sun.v.x = y[4 * i + 2]
            sun.v.y = y[4 * i + 3]
        self.p.pos.x = y[-4]
        self.p.pos.y = y[-3]
        self.p.v.x = y[-2]
        self.p.v.y = y[-1]

    def update_vw(self):
        theta = num2angle(self.p.v.y / abs(self.p.v), self.p.v.x / abs(self.p.v))
        # theta = self.time / 8 % 2 * math.pi
        # rect = pygame.draw.rect(self.screen, (0, 0, 70), self.view_window)

        img = np.zeros([self.view_window[2], self.view_window[3]])

        for s in self.star_map:
            if theta - math.pi / 4 < s[0] < theta + math.pi / 4:
                try:
                    img[int((s[0] - theta + math.pi / 4) / (math.pi / 2) * self.view_window[2]),
                    int(s[1])] = s[2]
                except Exception as e:
                    pass

        for sun in self.sun_list:
            d = distance(self.p.pos, sun.pos)
            alpha = num2angle((sun.pos.y - self.p.pos.y) / d, (sun.pos.x - self.p.pos.x) / d)
            alpha -= math.pi * 2

            for i in range(3):
                if theta - math.pi / 3 < alpha < theta + math.pi / 3:
                    x = alpha - theta + math.pi / 4
                    x /= math.pi / 2
                    x *= self.view_window[2]
                    y = self.view_window[3] / 2

                    if d > 125:
                        if x < 0:
                            continue
                        try:
                            img[int(x), int(y)] = 255
                            img[int(x), int(y + 1)] = 255
                            img[int(x + 1), int(y)] = 255
                            img[int(x + 1), int(y + 1)] = 255
                            img[int(x), int(y - 1)] = 255
                            img[int(x + 1), int(y - 1)] = 255
                            img[int(x - 1), int(y)] = 255
                            img[int(x - 1), int(y + 1)] = 255
                            img[int(x - 1), int(y - 1)] = 255
                            img[int(x), int(y + 2)] = 255
                            img[int(x), int(y - 2)] = 255
                            img[int(x + 2), int(y)] = 255
                            img[int(x - 2), int(y)] = 255
                        except:
                            pass
                    else:
                        r = sun.r * self.view_window[2] / (math.pi / 2 * d)
                        # res = np.ones([self.view_window[2], self.view_window[3]], dtype=np.uint8)
                        # res[int(x), int(y)] = 0
                        # res = cv2.distanceTransform(res, cv2.DIST_L2, 1)
                        # print(res)
                        for i in range(self.view_window[2]):
                            for j in range(self.view_window[3]):
                                try:
                                    d = ((i - x) ** 2 + (j - y) ** 2) ** 0.5 - r
                                    if d<0:
                                        num = 255
                                    else:
                                        num = int(255 * 0.8 ** d)
                                    num = 255 if num > 255 else num
                                    img[i, j] = num if num > img[i, j] else img[i, j]
                                except:
                                    pass
                    break

                alpha += math.pi * 2

        img = np.stack((img,) * 3, axis=-1)
        # cv2.imshow('img', img)
        self.screen.blit(pygame.surfarray.make_surface(img), (self.view_window[0], self.view_window[1]))

    def reset_sl(self):
        self.state_l = []
        for sun in self.sun_list:
            self.state_l.append(sun.pos.x)
            self.state_l.append(sun.pos.y)
            self.state_l.append(sun.v.x)
            self.state_l.append(sun.v.y)
        self.state_l.append(self.p.pos.x)
        self.state_l.append(self.p.pos.y)
        self.state_l.append(self.p.v.x)
        self.state_l.append(self.p.v.y)

    def fusion(self):
        if len(self.sun_list) > 1:
            for i in range(len(self.sun_list)):
                j = (i + 1) % len(self.sun_list)
                num = min(i, j)
                ma = max(i, j)
                if distance(self.sun_list[i].pos, self.sun_list[j].pos) <= 1:
                    j = self.sun_list[j]
                    i = self.sun_list[i]
                    self.sun_list[num] = Sun((i.pos.x, i.pos.y),
                                             (i.v * i.m + j.v * j.m) / (i.m + j.m),
                                             (i.m + j.m) * 0.98, 6000, num)
                    self.sun_list.pop(ma)
                    self.reset_sl()
                    break

    def gm_key_down(self, event):
        if event.key == pygame.K_UP:
            self.star_list[self.choosed].pos.y -= 5
        elif event.key == pygame.K_DOWN:
            self.star_list[self.choosed].pos.y += 5
        elif event.key == pygame.K_LEFT:
            self.star_list[self.choosed].pos.x -= 5
        elif event.key == pygame.K_RIGHT:
            self.star_list[self.choosed].pos.x += 5
        elif event.key == pygame.K_q:
            self.choosed = (self.choosed + 1) % len(self.star_list)
        elif event.key == pygame.K_w:
            self.star_list[self.choosed].v.y -= 1
        elif event.key == pygame.K_s:
            self.star_list[self.choosed].v.y += 1
        elif event.key == pygame.K_a:
            self.star_list[self.choosed].v.x -= 1
        elif event.key == pygame.K_d:
            self.star_list[self.choosed].v.x += 1
        elif event.key == pygame.K_DELETE:
            try:
                self.sun_list.remove(self.star_list[self.choosed])
            except:
                pass

    def start(self):
        while True:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    self.event_list.append(event.key)
                    if len(self.event_list) > 10:
                        self.event_list.pop(0)
                    if event.key == pygame.K_ESCAPE:
                        sys.exit()
                    elif event.key == pygame.K_0:
                        self.reset()
                        print('reset')
                    elif event.key == pygame.K_1:
                        self.show = not self.show
                    elif event.key == pygame.K_2:
                        self.show_trace = not self.show_trace
                    elif event.key == pygame.K_3:
                        self.show_vw = not self.show_vw
                    elif event.key == pygame.K_4:
                        self.gm = False
                        self.reset_sl()
                    elif self.event_list == self.gm_code:
                        self.gm = not self.gm
                        if not self.gm:
                            self.reset_sl()
                        self.star_list = []
                        self.choosed = 0
                        for s in self.sun_list:
                            self.star_list.append(s)
                        self.star_list.append(self.p)
                    if self.gm:
                        self.gm_key_down(event)

            if not self.gm:
                for i in range(self.per_frame):
                    try:
                        self.update()
                        update_p_temperature(self.p, self.sun_list, self.dt * 10)
                    except ValueError:
                        self.show_text(f"第{self.turn}号文明在无边的火海中毁灭了 文明的种子仍在")
                    if (i + 1) % 3 == 0:
                        self.fusion()

                self.time += self.dt * self.per_frame * 10

                if self.p.temp < 20:
                    self.cold += 0.1
                    if self.cold >= 25:
                        self.show_text(f"第{self.turn}号文明在无尽的严寒中毁灭了 文明的种子仍在")
                else:
                    self.cold = 0

                for i, sun in enumerate(self.sun_list):
                    self.trace[sun.i].append(tuple((sun.pos.x, sun.pos.y)))
                self.trace[-1].append(tuple((self.p.pos.x, self.p.pos.y)))

            self.screen.fill((0, 0, 0))
            self.screen.blit(self.text, (0, 0))

            for i, sun in enumerate(self.sun_list):
                self.screen.blit(sun.img, (sun.pos.x - 50, sun.pos.y - 50))

            if self.show_trace:
                for i in range(len(self.trace)):
                    if len(self.trace[i]) > 1:
                        pygame.draw.aalines(self.screen, self.colors[i], False, self.trace[i], 1)

            if self.gm:
                pos = self.star_list[self.choosed].pos
                pygame.draw.aalines(self.screen, (255, 255, 255), True,
                                    [(pos.x - 15, pos.y - 15), (pos.x - 15, pos.y + 15), (pos.x + 15, pos.y + 15),
                                     (pos.x + 15, pos.y - 15)])

            if self.show:
                for i, sun in enumerate(self.sun_list + [self.p]):
                    self.screen.blit(self.f_s.render(f"sun{i + 1}:{str(sun.pos)}", True, (200, 200, 200)),
                                     (0, 30 + 2 * i * 20))
                    self.screen.blit(self.f_s.render(f"v:{abs(sun.v)}",
                                                     True, (200, 200, 200)),
                                     (0, 30 + 2 * i * 20 + 20))
                    self.screen.blit(self.f_s.render(f"{i+1}",
                                                     True, (0, 0, 0)),
                                     (sun.pos.x - 4, sun.pos.y - 7))

            pygame.draw.circle(self.screen, (0, 0, 255), (self.p.pos.x, self.p.pos.y), 2)
            arrow(self.screen, (255, 255, 255), (255, 255, 255), (self.p.pos.x, self.p.pos.y),
                  (self.p.pos.x + self.p.v.x * 0.5, self.p.pos.y + self.p.v.y * 0.5), 3)

            if self.show_vw:
                self.update_vw()

            for sun in self.sun_list:
                if sun.pos.x < - self.width * 0.5 or sun.pos.x > self.width * 1.5 or\
                        sun.pos.y < -self.height * 0.5 or sun.pos.y > self.height * 1.5:
                    self.sun_list.remove(sun)
                    self.reset_sl()
                    break

            if self.p.pos.x < 0 or self.p.pos.x > self.width or self.p.pos.y < 0 or self.p.pos.y > self.height:
                self.reset()
            if self.time >= 150:
                self.screen.fill((0, 0, 0))
                text = self.f.render(f"恒纪元的梦，在第{self.turn}号文明终于成为了现实！种子：{self.seed}", True,
                                     (255, 255, 255), (0, 0, 0))
                self.screen.blit(text, (self.width / 2 - text.get_width() / 2, self.height
                                        / 2 - text.get_height() / 2))

                pygame.display.update()
                t.sleep(3)
                self.reset()
            self.text = self.f.render(f"第{self.turn}轮文明 持续时间：{round(self.time, 2):.2f} 种子:{self.seed}"
                                      f" 温度:{round(self.p.temp, 1)}K", True,
                                      (255, 255, 255), (0, 0, 0))

            if round(self.time, 3) % 2 == 0:
                new_x = self.p.pos.x
                new_y = self.p.pos.y
                new_x -= self.width / 2
                new_y -= self.height / 2
                r = Vector2D(new_x, new_y)
                self.p.pos -= r
                for sun in self.sun_list:
                    sun.pos -= r
                for tr in self.trace:
                    for i in range(len(tr)):
                        tr[i] = (tr[i][0] - r.x, tr[i][1] - r.y)

                self.reset_sl()

            if self.p.temp > 3000:
                self.show_text(f"第{self.turn}号文明在无边的火海中毁灭了 文明的种子仍在")
            pygame.display.update()

    def show_text(self, text):
        self.screen.fill((0, 0, 0))
        text = self.f.render(text, True,
                             (255, 255, 255), (0, 0, 0))
        self.screen.blit(text, (self.width / 2 - text.get_width() / 2, self.height
                                / 2 - text.get_height() / 2))

        pygame.display.update()
        t.sleep(3)
        self.reset()


if __name__ == '__main__':
    ai = Mainboard()
    ai.start()
