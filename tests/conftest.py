# adds the project root to sys.path so tests can import src.* and the category packages (training, evaluation, ...)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
