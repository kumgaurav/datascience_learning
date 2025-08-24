conda env remove -n stocks3
conda clean -a -y
conda info --envs   # note the full path for stocks3 if present
rm -rf /Users/gaurav/miniconda3/envs/stocks3
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
find . -name ".pytest_cache" -type d -prune -exec rm -rf {} +
find . -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} +
conda env create -f stocks3_env.yaml
conda run -n stocks3 python tests/tensor_flow_test.py | cat
conda run -n stocks3 python -c "import sqlalchemy, joblib, optuna, psycopg2, pymysql; print('ok')"