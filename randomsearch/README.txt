conda create -n rs_env python
conda activate rs_env
pip install scikit-learn
python rs.py
pip install mpi4py
mpiexec -n 4 --oversubscribe python rs-mpif.py
