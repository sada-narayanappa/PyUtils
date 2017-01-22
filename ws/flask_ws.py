### #!/anaconda/bin/python

from flask import Flask, jsonify, abort, request, make_response, url_for
from flask_restful import Resource, Api
from json import dumps
import sys, os, importlib, inspect
import wsUtils

app = Flask(__name__)

def echo():
    tags = [ t for t in request.args ];
    reqs =  str(request.json)
    t = str(tags)  + "\n" + reqs + "\n"
    print (t)
    return t

def process():
    pass;

@app.route('/ws', methods=['POST', 'GET', 'OPTIONS', 'HEAD'])
def index():
    if (not request.json or len(request.args) <= 0):
        return "Version 1.0\n";
    r = echo();
    return r;

@app.route('/run', methods=['POST', 'GET', 'OPTIONS', 'HEAD'])
def run():
    ret = wsUtils.Run('test', 'GetCustomer');
    return ret;

def start():
    port = 8500
    print( "Starting at port: ", port)
    app.run(debug=True, host="0.0.0.0", port=port)

if __name__ == '__main__':
    start();

