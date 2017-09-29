#!/usr/local/bin/python
#--------------------------------------------------------------------------
# To run type flask_ws <port: default: 8500) 
# How to Run
from flask import Flask
from flask import request
from flask import Response
import sys, os, importlib, inspect
from Jupytils.DBUtils import DBUtils
import traceback

app = Flask(__name__)
#--------------------------------------------------------------------------
@app.route('/ds/echo', methods=['POST', 'GET', 'OPTIONS', 'HEAD'])
def echo():
    tags = [t+"="+request.args[t] for t in request.args ]
    reqs =  str(request.json)
    t = str(tags)  + "\n" + reqs + "\n"
    return t
#--------------------------------------------------------------------------
@app.route('/ds/q/<qid>/', methods=['POST', 'GET', 'OPTIONS', 'HEAD'])
@app.route('/ds/q/<type>/<qid>/', methods=['POST', 'GET', 'OPTIONS', 'HEAD'])
def q(type = 'html', qid=''):
    if ( qid in db.QCACHE ):
        q1 = db.QCACHE[qid]
    else:
        q1 = qid;

    try: 
        q2, tags = DBUtils.prepare(q1, request.args);
        if (type not in ['html', 'text', 'json', 'csv'] ):
            type = 'html';

        df = db.execQ(q2)

        if (type == 'html'):
            r = q2 + "<br/><br/>"
            r = r + df.to_html();
            return Response(r, mimetype='text/html');
        elif (type == 'text' or type == 'csv' ):
            r = df.to_csv(index=False);
            return Response(r, mimetype='text');
        else:
            r = df.to_json();
            return Response(r, mimetype='text/json');
    except Exception as E:
        #E = sys.exc_info()[0]
        traceback.print_exc(file=sys.stdout)
        r = "Exception: " + qid + ":" + type + "\n\n" + q1 + "\n\n" + q2 + "\n\n"+ str(E);
        return Response(r, mimetype="text");

#--------------------------------------------------------------------------
@app.route('/ds/dump', methods=['POST', 'GET', 'OPTIONS', 'HEAD'])
def dump():
    ret = db.SQL
    return  Response(ret, mimetype='text')
#--------------------------------------------------------------------------
@app.route('/ds', methods=['POST', 'GET', 'OPTIONS', 'HEAD'])
def ws():
    defr = '''Data Service: Version 1.0

Available URLS are: /ds/dump, /ds/q , /ds/echo
'''
    if (len(request.args) <= 0):
        return Response(defr, mimetype='text');
#--------------------------------------------------------------------------
'''
Help:
flask_ws port connection 
'''
def start(sqlText=None, conn=None):
    global db
    db = DBUtils(conn = conn);
    db.load(file=sqlText);
    
    port = 8500 if len(sys.argv) < 2 else int(sys.argv[1])
    print( "Starting at port: ", port)
    app.run(debug=True, host="0.0.0.0", port=port)

#--------------------------------------------------------------------------
if __name__ == '__main__':
    SQL_TXT = "SQL.txt";
    conn='postgresql://postgres:postgres@localhost:5400/SCHASDB'

    start(SQL_TXT, conn, );

