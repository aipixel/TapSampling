set -euo pipefail

pip install wheel cmake==3.18.4.post1

echo "install tacto"
cd calvin_env/tacto
pip install -e .

echo "install calvin_env"
cd ..
pip install -e .

echo "install calvin_models"
cd ../calvin_models

# setuptools 75.3.0 -> 57.5.0
pip install setuptools==57.5.0
pip install --no-build-isolation MulticoreTSNE
pip install --no-build-isolation pyhash
pip install -e .

pip install setuptools==75.3.0

