import sys
import os


def setup_path():
    import sys
    import os
    third_party_path = os.path.abspath("./src")
    if third_party_path not in sys.path:
        sys.path.append(third_party_path)


setup_path()
