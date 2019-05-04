import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand

class CrowdSimulation(object):  
    def __init__(self, N=10, x_length=20.0, y_length=10.0,
                 step_length=0.8, step_angle=3.14, sigma=1,
                 reverse_rate=0.1):
        self.N           = N
        self.x_length    = x_length
        self.y_length    = y_length
        self.step_length = step_length
        self.step_angle  = step_angle
        self.sigma       = sigma
        self.reverse_rate = reverse_rate


    def setup_positions(self):
        interval = self.sigma*1.01
        n_yside = int(self.y_length/ interval)
        xs = np.arange(self.N) // n_yside
        xs = (xs + 0.5) / max(xs+1) * self.x_length
        ys = np.arange(self.N) % n_yside
        ys = (ys + 0.5) / max(ys+1) * self.y_length
        self.positions  = np.c_[xs,ys]
        n_reverse = int(self.N * self.reverse_rate)
        directions = np.zeros(self.N, dtype=np.float)
        directions[:n_reverse] = np.pi
        np.random.shuffle(directions)
        self.directions = directions
                
    def step(self):
        counter = [0,0]
        for i in range(self.N):
            step_vec = self.calc_onestep(theta0=self.directions [i])
            nears    = self.calc_nearlist(i)                
            t        = self.calc_collision(i,nears,step_vec)
            x0 = self.positions[i][0]
            self.move_agent(i,step_vec*t)
            x1 = self.positions[i][0]
            boarderline = self.x_length / 2
            direction   = self.directions[i]
            if (x0<boarderline) and (boarderline<x1) and (direction<np.pi/2):
                counter[0] += 1
            elif (x1<boarderline) and (boarderline<x0) and (direction>np.pi/2):
                counter[1] += 1
        return counter

    def calc_onestep(self, theta0=0.0):
        theta = (rand() - 0.5) * self.step_angle + theta0
        sx    = np.cos(theta) * self.step_length
        sy    = np.sin(theta) * self.step_length
        return np.array([sx, sy])
    
    def calc_nearlist(self,i):
        near_length = self.step_length + self.sigma
        allp  = np.r_[self.positions[:i],self.positions[i+1:]]
        xl,yl = self.x_length, self.y_length
        allp  = np.r_[allp, allp+np.array([ xl, 0]), allp-np.array([xl, 0])]
        x,y   = self.positions[i]
        nears_list = []
        for pos in allp:
            xj, yj = pos
            d = max(abs(x-xj), abs(y-yj))
            if d < near_length:
                nears_list.append([xj,yj])
        return np.array(nears_list)
        
    def calc_collision(self,i,nears,step_vec):
        t=1.0
        x,y   = self.positions[i]
        sx,sy = step_vec
        for xj,yj in nears:
            dx,dy = xj-x, yj-y
            a = sx*sx + sy*sy
            b = sx*dx + sy*dy
            c = dx*dx + dy*dy
            if (b>0) and (self.sigma>c-b*b/a):
                new_t = b/a - np.sqrt((self.sigma+b*b/a-c)/a)
                t = min(t,new_t)
        t *= rand()
        return t
    
    def move_agent(self,i,step_vec):
        new_x,new_y = self.positions[i] + step_vec
        if new_x > self.x_length:
            new_x = new_x - self.x_length
        if new_x < 0:
            new_x = self.x_length + new_x
        if new_y > self.y_length:
            new_y = self.y_length
        if new_y < 0:
            new_y = 0
        self.positions[i] = np.array([new_x, new_y])

        
if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.animation as animation
    n_step = 300

    env = CrowdSimulation()
    env.setup_positions()

    fig = plt.figure(figsize=(3,3))
    ax1 = fig.add_subplot(211)
    ax1.set_xlim(0, env.x_length)
    ax1.set_ylim(0, env.y_length)
    ax2 = fig.add_subplot(212)
    ax2.set_xlim(0, n_step)
    ims = []
    counter  = [0,0]
    counters = []
    print('* Run Simulation !')
    for _ in tqdm(range(n_step)):
        xs,ys  = env.positions.T
        isright= env.directions < np.pi/2
        isleft = np.invert(isright)
        im11,  = ax1.plot(xs[isright], ys[isright], 'ro', ms=8, alpha=0.3)
        im12,  = ax1.plot(xs[isleft],  ys[isleft],  'bo', ms=8, alpha=0.3)
        counters.append(counter)
        counters0=np.array(counters)[:,0]
        im21,  = ax2.plot(counters0, 'r-', alpha=0.1)
        im22,  = ax2.plot(np.convolve(counters0,np.ones(30)/30, mode='same'), 'r-')
        counters1=np.array(counters)[:,1]
        im23,  = ax2.plot(counters1, 'b-', alpha=0.1)
        im24,  = ax2.plot(np.convolve(counters1,np.ones(30)/30, mode='same'), 'b-')
        ims.append([im11,im12,im21,im22,im23,im24])
        counter = env.step()
    
    print('* Make Animation !')
    ani = animation.ArtistAnimation(fig, ims, interval=5,blit=True)
    ani.save('crowd-simulation.gif', writer="imagemagick",fps=60)
    ani.save('crowd-simulation.mp4', writer="ffmpeg")
    plt.show()
    
    