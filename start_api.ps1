$ProjectPath = "/mnt/c/Users/sandeep/OneDrive/Documents/Ai_Scada"

wsl bash -lc "cd '$ProjectPath' && source /home/sandeep/miniforge3/etc/profile.d/conda.sh && conda activate rapids-gpu && python -m uvicorn scripts.api.main:app --host 0.0.0.0 --port 8000"