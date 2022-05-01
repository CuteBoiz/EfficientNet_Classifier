# EficientNet_Classifier

## <div align="center"> I. TRAIN. </div>

```sh
python3 train.py
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
```

- **label.txt:**
```sh
dog
cat
```


## <div align="center"> II. RUN TEST SET AND EXPORT RESULT TO EXCEL FILE. </div>

```sh
python3 test.py
```

- **Arguments:**

| Arguments | Type | Default | Help
|-----------|------|---------|------


## <div align="center"> III. INFERENCE MODEL ON IMAGE/VIDEO/IMAGES FOLDER. </div>

```sh
python3 infer.py
```

- **Arguments:**

| Arguments | Type | Default | Help
|-----------|------|---------|------


## <div align="center"> IV. EXPORT ONNX. </div>

```sh
python3 export.py
```

- **Arguments:**

| Arguments | Type | Default | Help
|-----------|------|---------|------
