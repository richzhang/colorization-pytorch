mkdir -p ./checkpoints/siggraph_retrained
MODEL_FILE=./checkpoints/siggraph_retrained/latest_net_G.pth
URL=http://colorization.eecs.berkeley.edu/siggraph/models/pytorch.pth

wget -N $URL -O $MODEL_FILE

mkdir -p ./checkpoints/siggraph_caffemodel
MODEL_FILE=./checkpoints/siggraph_caffemodel/latest_net_G.pth
URL=http://colorization.eecs.berkeley.edu/siggraph/models/caffemodel.pth

wget -N $URL -O $MODEL_FILE
