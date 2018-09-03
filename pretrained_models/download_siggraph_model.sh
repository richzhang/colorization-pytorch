mkdir -p ./checkpoints/siggraph_pretrained
MODEL_FILE=./checkpoints/siggraph_pretrained/latest_net_G.pth
URL=[[FILL/THIS/IN]]

wget -N $URL -O $MODEL_FILE


