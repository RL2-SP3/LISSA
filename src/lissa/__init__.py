# RL2-SP3

#⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⣠⣾⠟⠛⠛⠷⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⡀⠀⠀⠀
# ⠀⠀⠀⠀⣰⡟⠁⠀⠀⠀⠀⠀⠛⢷⣄⢀⣠⣤⡶⠾⠟⠛⠛⠛⠛⠛⠛⠻⢷⡄
# ⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⣟⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢘⣿
# ⠀⠀⠀⢰⣿⠀⠀⢰⣆⠀⠀⠀⠀⠀⠈⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠏
# ⠀⢀⣤⣼⣿⠶⣦⡄⠛⢷⣤⣤⣀⣀⣴⡿⢶⣦⣤⣤⣤⣤⣶⣶⡶⠾⠟⠛⠁⠀
# ⢰⣟⠁⠉⣁⠀⢼⣇⠀⠀⠀⠉⠉⠉⠁⠀⠀⠀⠀⠙⠻⢶⣄⠀⠀⠀⠀⠀⠀⠀
# ⠀⢹⡶⠈⠛⣀⠀⣹⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⣦⠀⠀⠀⠀⠀⠀
# ⠀⠘⢿⣶⡾⠛⠛⠋⠀⠀⠀⠀⢀⣀⣠⣤⣤⣤⣀⡀⠀⠀⠀⢻⡇⠀⠀⠀⠀⠀
# ⠀⠀⣿⠃⠀⠀⠀⠀⣀⣤⡶⠟⠛⠋⠉⠉⠉⠉⠉⠛⠿⣦⡀⢸⡇⠀⠀⠀⠀⠀
# ⠀⠸⣿⠀⠀⠀⣠⡾⠛⡁⠀⠀⠀⠀⠀⠀⠀⠀⣾⣷⠀⢹⣧⣿⠁⠀⠀⠀⠀⠀
# ⠀⠀⢻⣆⠀⢰⡟⠁⢸⣿⠀⠀⠀⠀⣴⡄⠀⠀⠉⠁⠀⣸⡿⠃⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠻⣦⣿⣇⠀⠈⠉⠀⠀⠀⢰⣶⡾⠇⠀⢀⣤⡾⠟⠁⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠈⠛⢿⣷⣤⣄⣀⣀⣀⣀⣠⣤⣴⡾⠟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠛⠛⠋⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

# Library Intended for Solving Special Assignments (LISSA)
# 
# Created for Equinor to ESP data processing (as it came)
# 
# Developed at EPIC by Paulo Yoshio Kuga


from .ml import *
from .pca import *
from .picture import *
from .processing import *

import locale


__all__ = ["ml","pca","picture","processing"]

version = "2025-01-31"

# Tenta mudar o locale para pt_BR (varia por sistema)
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except locale.Error:
    print("Locale pt_BR não disponível no sistema.")
