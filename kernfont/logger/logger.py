import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
)

# Imposta il livello di logging per il modulo _internal.py su un livello superiore (es. WARNING) per disabilitare i log
# logging.getLogger('internal.py').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
