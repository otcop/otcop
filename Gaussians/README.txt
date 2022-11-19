Code to get the results in section 4.1-4.2 of the paper "OTCOP: Learning optimal transport maps via constraint optimizations"
Uncomment each line to run
#python main.py
#python run_high_dim.py --data 78DGaussian  --length 100 --learning_rate 0.0001 --method AL
#python run_high_dim.py --data 78DGaussian  --length 100 --learning_rate 0.0001 --method SP
#python run_high_dim.py --data 78DGaussian  --length 100 --learning_rate 0.0001 --method ADMM
#python run_high_dim.py --data 784DGaussian  --length 10 --learning_rate 0.0001 --method SP
#python run_high_dim.py --data 784DGaussian  --length 10 --learning_rate 0.0001 --method AL
#python run_high_dim.py --data 784DGaussian  --length 10 --learning_rate 0.0001 --method ADMM


#python run_gaussianmixture.py --data 2DGaussian  --length 10 --learning_rate 0.0001 --method SP
#python run_gaussianmixture.py --data 2DGaussian  --length 10 --learning_rate 0.0001 --method AL
#python run_gaussianmixture.py --data 2DGaussian  --length 10 --learning_rate 0.0001 --method ADMM

#python run_gaussianmixture.py --data 2DMixture  --length 10 --learning_rate 0.0001 --method SP_no
#python run_gaussianmixture.py --data 2DMixture  --length 10 --learning_rate 0.0001 --method SP
#python run_gaussianmixture.py --data 2DMixture  --length 10 --learning_rate 0.0001 --method SP_no
#python run_gaussianmixture.py --data 2DMixture  --length 10 --learning_rate 0.0001 --method AL
#python run_gaussianmixture.py --data 2DMixture  --length 10 --learning_rate 0.0001 --method QP
#python run_gaussianmixture.py --data 2DMixture  --length 10 --learning_rate 0.0001 --method ADMM
