import json

from flask import Flask, render_template
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    N=1000
    xs = np.arange(N)
    ys = np.random.randn(N).cumsum()
    points = [[x,y] for x,y in zip(xs,ys)]
    data = {'points':points}
    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)