# src/__init__.py

# Denne filen gjør at 'src' mappen kan behandles som et Python-pakke
# Den kan være tom, eller du kan samle ofte brukte imports her:

# Eksempel på convenience-imports:
# from .data_pipeline import make_dataset, load_labels, parse_pair
# from .models import build_generator, build_discriminator
# from .train_gan import train
# from .evaluate import evaluate
# from .utils import load_config, ensure_dir, normalize_img

# Hvis du bruker imports over, kan du gjøre:
# import src
# src.train()
# i stedet for:
# from src.train_gan import train
# train()

# Mangler / å vurdere:
# - Definere __all__ med navn på funksjoner/klasser som skal eksporteres
# - Oppdatere når du legger til nye moduler i src/
# - Sørge for at navnet på pakkemappen (src) matcher importene i resten av koden
