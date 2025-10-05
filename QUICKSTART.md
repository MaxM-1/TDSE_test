# Quick Start Guide

## Activating the Virtual Environment

### Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

### Windows Command Prompt:
```cmd
.\venv\Scripts\activate.bat
```

### Linux/Mac:
```bash
source venv/bin/activate
```

## Deactivating
```
deactivate
```

## Installing Additional Packages
With the virtual environment activated:
```
pip install package-name
```

## Running Examples
```
python examples\quantum_tunneling.py
python examples\harmonic_oscillator.py
python examples\free_particle.py
python examples\potential_well.py
python examples\double_slit.py
```

## Running Tests
```
pytest tests/ -v
```

## Package Installation Status
✅ NumPy 2.3.3
✅ SciPy 1.16.2
✅ Matplotlib 3.10.6
✅ pytest 8.4.2
✅ pytest-cov 7.0.0
✅ tdse-solver 0.1.0 (installed in editable mode)
