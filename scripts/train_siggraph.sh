
# Train classification network first
python train.py --name siggraph_class --sample_p 1.0 --niter 15 --niter_decay 0 --classification

# Train regression model (with color hints)
mkdir ./checkpoints/siggraph_reg
cp ./checkpoints/siggraph_class/latest_net_G.pth ./checkpoints/siggraph_reg/
python train.py --name siggraph_reg --sample_p .125 --niter 10 --niter_decay 0 --lr 0.00001 --load_model

# Turn down learning rate to 1e-6
mkdir ./checkpoints/siggraph_reg2
cp ./checkpoints/siggraph_reg/latest_net_G.pth ./checkpoints/siggraph_reg2/
python train.py --name siggraph_reg2 --sample_p .125 --niter 5 --niter_decay 0 --lr 0.000001 --load_model
