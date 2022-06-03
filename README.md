# EficientNet_Classifier

## <div align="center"> I. Train EfficientNet Classifier. </div>

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

## <div align="center"> II. Run test set and visualize result onto excel file. </div>

```sh
python3 test.py
```

- **Arguments:**

| Arguments | Type | Default | Help
|-----------|------|---------|------


## <div align="center"> III. Inference on image/video/images folder. </div>

```sh
python3 infer.py
```

- **Arguments:**

| Arguments | Type | Default | Help
|-----------|------|---------|------


## <div align="center"> IV. Export to ONNX engine. </div>

```sh
python3 export.py
```

- **Arguments:**

| Arguments | Type | Default | Help
|-----------|------|---------|------
