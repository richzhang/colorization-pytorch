mkdir -p ./checkpoints/siggraph_pretrained
MODEL_FILE=./checkpoints/siggraph_pretrained/latest_net_G.pth
URL=http://colorization.eecs.berkeley.edu/siggraph/models/pytorch.pth

wget -N $URL -O $MODEL_FILE


