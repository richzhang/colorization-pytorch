
# Train classification network first
python train.py --name siggraph_class --sample_p 1.0 --niter 15 --niter_decay 0 --classification

# Train regression model (with color hints)
mkdir ./checkpoints/reg
cp ./checkpoints/class/latest_net_G.pth ./checkpoints/reg/
python train.py --name siggraph_reg --sample_p .125 --niter 10 --niter_decay 0 --lr 0.00001 --load_model
