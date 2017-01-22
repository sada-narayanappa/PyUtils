import io, nbformat, subprocess;
import sys, os, importlib, inspect
from default_ws_config import *
import wsUtils

#port=8501
#nbs = [
    #"../../../notebooks/PyUtils/ws/wsex1.ipynb",
    #"../../../notebooks/PyUtils/ws/wsex2.ipynb",
#]
def deploy():
    global notebooks
    if ( len(sys.argv) > 1 ):
        print ("Importing from ", sys.argv[1]);
        im = importlib.import_module(sys.argv[1])
        notebooks = im.notebooks

    merge_notebooks(notebooks, 'wsex.ipynb', False);

def run():
    OPTS='--KernelGatewayApp.api="kernel_gateway.notebook_http" --KernelGatewayApp.prespawn_count=4'
    PORT='--KernelGatewayApp.port=8501'
    URI='../../../notebooks/PyUtils/ws/wsex.ipynb'
    URI1='--KernelGatewayApp.seed_uri='+URI

    p = subprocess.Popen(['jupyter', 'kernelgateway', OPTS, PORT, URI1] )
    p = subprocess.Popen("./kg.sh" )
    #p.wait()

def merge_notebooks(filenames, outf = None, printO=False):
    merged = None
    for fname in filenames:
        print("Merging ", fname, " ...");
        with io.open(fname, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        if merged is None:
            merged = nb
        else:
            # TODO: add an optional marker between joined notebooks
            # like an horizontal rule, for example, or some other arbitrary
            # (user specified) markdown cell)
            merged.cells.extend(nb.cells)
    if not hasattr(merged.metadata, 'name'):
        merged.metadata.name = ''
    merged.metadata.name += "_merged"

    if (printO):
        print(nbformat.writes(merged))
    if (outf is not None):
        print("Writing to file ", outf, " ...")
        with( io.open(outf, "w", encoding='utf-8')) as mf:
            mf.write(nbformat.writes(merged))


# Command line merge tool
def cmd():
    notebooks = sys.argv[1:]
    if not notebooks:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
        
    merge_notebooks(notebooks)

if __name__ == '__main__':
    deploy();
    #run();
