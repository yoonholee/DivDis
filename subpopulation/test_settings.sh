# Quickly check whether all settings run properly

python run_expt.py --debug --setting WBIRDS
python run_expt.py --debug --setting MultiNLI
python run_expt.py --debug --setting CELEBA_1
# python run_expt.py --debug --setting CELEBA_2
# python run_expt.py --debug --setting CELEBA_3
# python run_expt.py --debug --setting CELEBA_4

python run_expt.py --debug --diversify --setting WBIRDS 
python run_expt.py --debug --diversify --setting MultiNLI
python run_expt.py --debug --diversify --setting CELEBA_1
# python run_expt.py --debug --diversify --setting CELEBA_2
# python run_expt.py --debug --diversify --setting CELEBA_3
# python run_expt.py --debug --diversify --setting CELEBA_4