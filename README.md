# GAN-BurstSelect
A Generative Adversarial Approach to Burst-Sequence Image Ranking



# aktiver venv om du bruker ett

python3 -m venv .venv        # hvis du ikke allerede har opprettet det
source .venv/bin/activate

pip install -r requirements.txt
python src/train_gan.py --config config.yaml
python src/evaluate.py  --config config.yaml
