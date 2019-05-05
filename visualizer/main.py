import json
from flask import Flask, render_template
import numpy as np

from simulation import CrowdSimulation

app = Flask(__name__)

env  = CrowdSimulation(N=100,step_length=0.1)
xlim = [0, env.x_length]
ylim = [0, env.y_length]

@app.route('/')
def index():
    n_step  = 100
    x0s = np.random.rand(n_step).cumsum()/100
    y0s = np.random.rand(n_step).cumsum()/100
    x1s = np.random.rand(n_step).cumsum()/100
    y1s = np.random.rand(n_step).cumsum()/100
    xs = [[x0,x1] for x0,x1 in np.c_[x0s,x1s]]
    ys = [[y0,y1] for y0,y1 in np.c_[y0s,y1s]]
    data = {'xs':xs,'ys':ys,'xlim':xlim,'ylim':ylim,'max_step':n_step}
    return render_template("index.html", data=data)

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    n_step = 1
    env.setup_positions()
    xs,ys = env.positions.T
    xs, ys = [list(xs)], [list(ys)]
    data = {'xs':xs,'ys':ys,'xlim':xlim,'ylim':ylim,'max_step':n_step}
    return render_template("index.html", data=data)


@app.route('/start', methods=['GET', 'POST'])
def start():
    n_step=1000
    xs,ys = env.positions.T
    xs, ys = [list(xs)], [list(ys)]
    for i in range(n_step):
        env.step()
        x,y = env.positions.T
        xs.append(list(x))
        ys.append(list(y))
    data = {'xs':xs,'ys':ys,'xlim':xlim,'ylim':ylim,'max_step':n_step}
    return render_template("index.html", data=data)



if __name__ == "__main__":
    app.run(debug=True)

