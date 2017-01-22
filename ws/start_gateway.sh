python deploy.py default_ws_config.py

export OPTS='--KernelGatewayApp.api="kernel_gateway.notebook_http" --KernelGatewayApp.prespawn_count=1'
export PORT='--KernelGatewayApp.port=8501'

export URI='../../../notebooks/PyUtils/ws/wsex.ipynb'
export URI1='--KernelGatewayApp.seed_uri='$URI

echo jupyter kernelgateway $OPTS $PORT $URI1 $URI2
jupyter kernelgateway $OPTS $PORT $URI1 $URI2
