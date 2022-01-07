# EficientNet_Classifier

## <div align="center"> I. TRAIN. </div>

```sh
python3 main.py train
```

- **Arguments:**

| Arguments | Type | Default | Help
|-----------|------|---------|------

- **Data path example:**
```sh
DataPath
|
|__train.txt
|__test.txt
|__valid.txt
|__label.txt
```
- **train/valid/test.txt files:**
```sh
"path/to/image" class_id
"path/to/image1.png" 0
"path/to/image2.jpg" 2
"path/to/image3.png" 1
...
```

- **label.txt:**
```sh
dog
cat
...
```


## <div align="center"> II. EXPORT EXCEL. </div>

```sh
python3 main.py export_excel
```

- **Arguments:**

| Arguments | Type | Default | Help
|-----------|------|---------|------


## <div align="center"> III. INFERENCE MOEL. </div>

```sh
python3 main.py infer
```

- **Arguments:**

| Arguments | Type | Default | Help
|-----------|------|---------|------


## <div align="center"> IV. EXPORT ONNX. </div>

```sh
python3 main.py export_onnx
```

- **Arguments:**

| Arguments | Type | Default | Help
|-----------|------|---------|------
