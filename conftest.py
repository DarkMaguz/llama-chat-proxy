import os
import sys

# Make the project root importable so `import llama_chat_proxy` works no matter
# how pytest is invoked.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
