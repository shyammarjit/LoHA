## Installation Steps

Conda env Create
```
conda create -y -n loha python=3.11
conda activate loha
```

Install diffuser from Our Space
```
cd LoHA 
pip install -e ".[torch]"
```

Install requirements 
```
pip install -r requirements.txt 
```

Train dreambooth_sdxl using script file
```
bash test_sdxl.sh
```

Generate images from the finetuned weights 
```
python generator.py
```